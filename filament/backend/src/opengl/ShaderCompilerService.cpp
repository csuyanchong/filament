/*
 * Copyright (C) 2023 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "ShaderCompilerService.h"

#include "BlobCacheKey.h"
#include "OpenGLBlobCache.h"
#include "OpenGLDriver.h"

#include <private/backend/BackendUtils.h>

#include <backend/Program.h>

#include <utils/compiler.h>
#include <utils/CString.h>
#include <utils/Log.h>
#include <utils/Systrace.h>

#include <string>
#include <string_view>
#include <variant>

namespace filament::backend {

using namespace utils;

// ------------------------------------------------------------------------------------------------

static void logCompilationError(utils::io::ostream& out,
        ShaderStage shaderType, const char* name,
        GLuint shaderId, CString const& sourceCode) noexcept;

static void logProgramLinkError(utils::io::ostream& out,
        const char* name, GLuint program) noexcept;

static inline std::string to_string(bool b) noexcept {
    return b ? "true" : "false";
}

static inline std::string to_string(int i) noexcept {
    return std::to_string(i);
}

static inline std::string to_string(float f) noexcept {
    return "float(" + std::to_string(f) + ")";
}

// ------------------------------------------------------------------------------------------------

struct ShaderCompilerService::ProgramToken {
    ProgramToken(ShaderCompilerService& compiler, utils::CString const& name) noexcept
            : compiler(compiler), name(name) {
    }
    ShaderCompilerService& compiler;
    utils::CString const& name;
    utils::FixedCapacityVector<std::pair<utils::CString, uint8_t>> attributes;
    std::array<utils::CString, Program::SHADER_TYPE_COUNT> shaderSourceCode;
    void* user = nullptr;
    struct {
        GLuint shaders[Program::SHADER_TYPE_COUNT] = {};
        GLuint program = 0;
    } gl; // 12 bytes
};

void ShaderCompilerService::setUserData(const program_token_t& token, void* user) noexcept {
    token->user = user;
}

void* ShaderCompilerService::getUserData(const program_token_t& token) noexcept {
    return token->user;
}

// ------------------------------------------------------------------------------------------------

ShaderCompilerService::ShaderCompilerService(OpenGLDriver& driver)
        : mDriver(driver),
          KHR_parallel_shader_compile(driver.getContext().ext.KHR_parallel_shader_compile) {
}

ShaderCompilerService::~ShaderCompilerService() noexcept = default;

ShaderCompilerService::program_token_t ShaderCompilerService::createProgram(
        utils::CString const& name, Program&& program) {
    auto& gl = mDriver.getContext();

    auto token = std::make_shared<ProgramToken>(*this, name);

    if (UTILS_UNLIKELY(gl.isES2())) {
        token->attributes = std::move(program.getAttributes());
    }

    BlobCacheKey key;
    token->gl.program = OpenGLBlobCache::retrieve(&key, mDriver.mPlatform, program);
    if (!token->gl.program) {
        // this cannot fail because we check compilation status after linking the program
        // shaders[] is filled with id of shader stages present.
        compileShaders(gl,
                std::move(program.getShadersSource()),
                program.getSpecializationConstants(),
                token->gl.shaders,
                token->shaderSourceCode);

        runAtNextTick(token, [this, token, key = std::move(key)]() {
            if (KHR_parallel_shader_compile) {
                // don't attempt to link this program if all shaders are not done compiling
                GLint status;
                if (token->gl.program) {
                    glGetProgramiv(token->gl.program, GL_COMPLETION_STATUS, &status);
                    if (status == GL_FALSE) {
                        return false;
                    }
                } else {
                    for (auto shader: token->gl.shaders) {
                        if (shader) {
                            glGetShaderiv(shader, GL_COMPLETION_STATUS, &status);
                            if (status == GL_FALSE) {
                                return false;
                            }
                        }
                    }
                }
            }

            if (!token->gl.program) {
                // link the program, this also cannot fail because status is checked later.
                token->gl.program = linkProgram(mDriver.getContext(), token);
                if (KHR_parallel_shader_compile) {
                    // wait until the link finishes...
                    return false;
                }
            }

            assert_invariant(token->gl.program);

            if (key) {
                // attempt to cache
                OpenGLBlobCache::insert(mDriver.mPlatform, key, token->gl.program);
            }

            return true;
        });
    }

    return token;
}

bool ShaderCompilerService::isProgramReady(
        const ShaderCompilerService::program_token_t& token) const noexcept {

    assert_invariant(token);

    if (!token->gl.program) {
        return false;
    }

    if (KHR_parallel_shader_compile) {
        GLint status = GL_FALSE;
        glGetProgramiv(token->gl.program, GL_COMPLETION_STATUS, &status);
        return (bool)status;
    }

    // If gl.program is set, this means the program was linked. Some drivers may defer the link
    // in which case we might block in getProgram() when we check the program status.
    // Unfortunately, this is nothing we can do about that.
    return bool(token->gl.program);
}

GLuint ShaderCompilerService::getProgram(ShaderCompilerService::program_token_t& token) {
    GLuint const program = initialize(token);
    assert_invariant(token == nullptr);
    assert_invariant(program);
    return program;
}

/* static*/ void ShaderCompilerService::terminate(program_token_t& token) {
    assert_invariant(token);

    token->compiler.cancelTickOp(token);

    for (GLuint& shader: token->gl.shaders) {
        if (shader) {
            if (token->gl.program) {
                glDetachShader(token->gl.program, shader);
            }
            glDeleteShader(shader);
            shader = 0;
        }
    }
    if (token->gl.program) {
        glDeleteProgram(token->gl.program);
    }

    token = nullptr;
}

void ShaderCompilerService::tick() {
    executeTickOps();
}

void ShaderCompilerService::notifyWhenAllProgramsAreReady(CallbackHandler* handler,
        CallbackHandler::Callback callback, void* user) {

    if (KHR_parallel_shader_compile) {
        // list all programs up to this point
        std::vector<program_token_t> tokens;
        for (auto& [token, _] : mRunAtNextTickOps) {
            if (token) {
                tokens.push_back(token);
            }
        }

        runAtNextTick(nullptr, [this, tokens = std::move(tokens), handler, user, callback]() {
            for (auto const& token : tokens) {
                assert_invariant(token);
                if (!isProgramReady(token)) {
                    // one of the program is not ready, try next time
                    return false;
                }
            }
            // all programs are ready, we can call the callbacks
            handler->post(user, callback);
            // and we're done
            return true;
        });

        return;
    }

    // we don't have KHR_parallel_shader_compile

    runAtNextTick(nullptr, [this, handler, user, callback]() {
        mDriver.scheduleCallback(handler, user, callback);
        return true;
    });

    // TODO: we could spread the compiles over several frames, the tick() below then is not
    //       needed here. We keep it for now as to not change the current beavior too much.
    // this will block until all programs are linked
    tick();
}

// ------------------------------------------------------------------------------------------------

GLuint ShaderCompilerService::initialize(ShaderCompilerService::program_token_t& token) noexcept {
    if (!token->gl.program) {
        // FIXME: with KHR_parallel_shader_compile we need to wait to get a program
        //  this can be forced by calling gl apis

        if (KHR_parallel_shader_compile) {
            // we force the program link -- which might stall, either here or below in
            // checkProgramStatus(), but we don't have a choice, we need to use the program now.
            token->gl.program = linkProgram(mDriver.getContext(), token);
        } else {
            // if we don't have a program yet, block until we get it.
            tick();
            // by this point we must have a GL program
            assert_invariant(token->gl.program);
        }
    }

    GLuint program = 0;

    // check status of program linking and shader compilation, logs error and free all resources
    // in case of error.
    bool const success = checkProgramStatus(token);
    if (UTILS_LIKELY(success)) {
        program = token->gl.program;
        // no need to keep the shaders around
        UTILS_NOUNROLL
        for (GLuint& shader: token->gl.shaders) {
            if (shader) {
                glDetachShader(program, shader);
                glDeleteShader(shader);
                shader = 0;
            }
        }
    }

    // and destroy all temporary init data
    token = nullptr;

    return program;
}


/*
 * Compile shaders in the ShaderSource. This cannot fail because compilation failures are not
 * checked until after the program is linked.
 * This always returns the GL shader IDs or zero a shader stage is not present.
 */
void ShaderCompilerService::compileShaders(OpenGLContext& context,
        Program::ShaderSource shadersSource,
        utils::FixedCapacityVector<Program::SpecializationConstant> const& specializationConstants,
        GLuint shaderIds[Program::SHADER_TYPE_COUNT],
        UTILS_UNUSED_IN_RELEASE std::array<CString, Program::SHADER_TYPE_COUNT>& outShaderSourceCode) noexcept {

    SYSTRACE_CALL();

    auto appendSpecConstantString = +[](std::string& s, Program::SpecializationConstant const& sc) {
        s += "#define SPIRV_CROSS_CONSTANT_ID_" + std::to_string(sc.id) + ' ';
        s += std::visit([](auto&& arg) { return to_string(arg); }, sc.value);
        s += '\n';
        return s;
    };

    std::string specializationConstantString;
    for (auto const& sc : specializationConstants) {
        appendSpecConstantString(specializationConstantString, sc);
    }
    if (!specializationConstantString.empty()) {
        specializationConstantString += '\n';
    }

        // build all shaders
            UTILS_NOUNROLL
    for (size_t i = 0; i < Program::SHADER_TYPE_COUNT; i++) {
        const ShaderStage stage = static_cast<ShaderStage>(i);
        GLenum glShaderType{};
        switch (stage) {
            case ShaderStage::VERTEX:
                glShaderType = GL_VERTEX_SHADER;
                break;
            case ShaderStage::FRAGMENT:
                glShaderType = GL_FRAGMENT_SHADER;
                break;
            case ShaderStage::COMPUTE:
#if defined(BACKEND_OPENGL_LEVEL_GLES31)
                glShaderType = GL_COMPUTE_SHADER;
#else
                continue;
#endif
                break;
        }

        if (UTILS_LIKELY(!shadersSource[i].empty())) {
            Program::ShaderBlob& shader = shadersSource[i];

            // remove GOOGLE_cpp_style_line_directive
            std::string_view const source = process_GOOGLE_cpp_style_line_directive(context,
                    reinterpret_cast<char*>(shader.data()), shader.size());

            // add support for ARB_shading_language_packing if needed
            auto const packingFunctions = process_ARB_shading_language_packing(context);

            // split shader source, so we can insert the specification constants and the packing functions
            auto const [prolog, body] = splitShaderSource(source);

            const std::array<const char*, 4> sources = {
                    prolog.data(),
                    specializationConstantString.c_str(),
                    packingFunctions.data(),
                    body.data()
            };

            const std::array<GLint, 4> lengths = {
                    (GLint)prolog.length(),
                    (GLint)specializationConstantString.length(),
                    (GLint)packingFunctions.length(),
                    (GLint)body.length() - 1 // null terminated
            };

            GLuint const shaderId = glCreateShader(glShaderType);
            glShaderSource(shaderId, sources.size(), sources.data(), lengths.data());
            glCompileShader(shaderId);

#ifndef NDEBUG
            // for debugging we return the original shader source (without the modifications we
            // made here), otherwise the line numbers wouldn't match.
            outShaderSourceCode[i] = { source.data(), source.length() };
#endif

            shaderIds[i] = shaderId;
        }
    }
}

// If usages of the Google-style line directive are present, remove them, as some
// drivers don't allow the quotation marks. This happens in-place.
std::string_view ShaderCompilerService::process_GOOGLE_cpp_style_line_directive(OpenGLContext& context,
        char* source, size_t len) noexcept {
    if (!context.ext.GOOGLE_cpp_style_line_directive) {
        if (UTILS_UNLIKELY(requestsGoogleLineDirectivesExtension({ source, len }))) {
            removeGoogleLineDirectives(source, len); // length is unaffected
        }
    }
    return { source, len };
}

// Tragically, OpenGL 4.1 doesn't support unpackHalf2x16 (appeared in 4.2) and
// macOS doesn't support GL_ARB_shading_language_packing
std::string_view ShaderCompilerService::process_ARB_shading_language_packing(OpenGLContext& context) noexcept {
    using namespace std::literals;
#ifdef BACKEND_OPENGL_VERSION_GL
    if (!context.isAtLeastGL<4, 2>() && !context.ext.ARB_shading_language_packing) {
        return R"(

// these don't handle denormals, NaNs or inf
float u16tofp32(highp uint v) {
    v <<= 16u;
    highp uint s = v & 0x80000000u;
    highp uint n = v & 0x7FFFFFFFu;
    highp uint nz = n == 0u ? 0u : 0xFFFFFFFF;
    return uintBitsToFloat(s | ((((n >> 3u) + (0x70u << 23))) & nz));
}
vec2 unpackHalf2x16(highp uint v) {
    return vec2(u16tofp32(v&0xFFFFu), u16tofp32(v>>16u));
}
uint fp32tou16(float val) {
    uint f32 = floatBitsToUint(val);
    uint f16 = 0u;
    uint sign = (f32 >> 16) & 0x8000u;
    int exponent = int((f32 >> 23) & 0xFFu) - 127;
    uint mantissa = f32 & 0x007FFFFFu;
    if (exponent > 15) {
        f16 = sign | (0x1Fu << 10);
    } else if (exponent > -15) {
        exponent += 15;
        mantissa >>= 13;
        f16 = sign | uint(exponent << 10) | mantissa;
    } else {
        f16 = sign;
    }
    return f16;
}
highp uint packHalf2x16(vec2 v) {
    highp uint x = fp32tou16(v.x);
    highp uint y = fp32tou16(v.y);
    return (y << 16) | x;
}
)"sv;
    }
#endif // BACKEND_OPENGL_VERSION_GL
    return ""sv;
}

// split shader source code in two, the first section goes from the start to the line after the
// last #extension, and the 2nd part goes from there to the end.
std::array<std::string_view, 2> ShaderCompilerService::splitShaderSource(std::string_view source) noexcept {
    auto start = source.find("#version");
    assert_invariant(start != std::string_view::npos);

    auto pos = source.rfind("\n#extension");
    if (pos == std::string_view::npos) {
        pos = start;
    } else {
        ++pos;
    }

    auto eol = source.find('\n', pos) + 1;
    assert_invariant(eol != std::string_view::npos);

    std::string_view const version = source.substr(start, eol - start);
    std::string_view const body = source.substr(version.length(), source.length() - version.length());
    return { version, body };
}

/*
 * Create a program from the given shader IDs and links it. This cannot fail because errors
 * are checked later. This always returns a valid GL program ID (which doesn't mean the
 * program itself is valid).
 */
GLuint ShaderCompilerService::linkProgram(OpenGLContext& context,
        program_token_t const& token) noexcept {

    SYSTRACE_CALL();

    GLuint const program = glCreateProgram();
    for (auto shader : token->gl.shaders) {
        if (shader) {
            glAttachShader(program, shader);
        }
    }

    if (UTILS_UNLIKELY(context.isES2())) {
        for (auto const& [ name, loc ] : token->attributes) {
            glBindAttribLocation(program, loc, name.c_str());
        }
    }

    glLinkProgram(program);

    return program;
}

// ------------------------------------------------------------------------------------------------

void ShaderCompilerService::runAtNextTick(const program_token_t& token, std::function<bool()> fn) noexcept {
    mRunAtNextTickOps.emplace_back(token, std::move(fn));
}

void ShaderCompilerService::cancelTickOp(program_token_t token) noexcept {
    // We do a linear search here, but this is rare, and we know the list is pretty small.
    auto pos = std::find_if(mRunAtNextTickOps.begin(), mRunAtNextTickOps.end(),
            [&](const auto& item) {
        return item.first == token;
    });
    if (pos != mRunAtNextTickOps.end()) {
        mRunAtNextTickOps.erase(pos);
    }
}

void ShaderCompilerService::executeTickOps() noexcept {
    auto& ops = mRunAtNextTickOps;
    auto it = ops.begin();
    while (it != ops.end()) {
        bool const remove = it->second();
        if (remove) {
            it = ops.erase(it);
        } else {
            ++it;
        }
    }
}

// ------------------------------------------------------------------------------------------------

/*
 * Checks a program link status and logs errors and frees resources on failure.
 * Returns true on success.
 */
bool ShaderCompilerService::checkProgramStatus(program_token_t const& token) noexcept {

    SYSTRACE_CALL();

    assert_invariant(token->gl.program);

    GLint status;
    glGetProgramiv(token->gl.program, GL_LINK_STATUS, &status);
    if (UTILS_LIKELY(status == GL_TRUE)) {
        return true;
    }

    // only if the link fails, we check the compilation status
    UTILS_NOUNROLL
    for (size_t i = 0; i < Program::SHADER_TYPE_COUNT; i++) {
        const ShaderStage type = static_cast<ShaderStage>(i);
        const GLuint shader = token->gl.shaders[i];
        if (shader) {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
            if (status != GL_TRUE) {
                logCompilationError(slog.e, type,
                        token->name.c_str_safe(), shader, token->shaderSourceCode[i]);
            }
            glDetachShader(token->gl.program, shader);
            glDeleteShader(shader);
            token->gl.shaders[i] = 0;
        }
    }
    // log the link error as well
    logProgramLinkError(slog.e, token->name.c_str_safe(), token->gl.program);
    glDeleteProgram(token->gl.program);
    token->gl.program = 0;
    return false;
}

UTILS_NOINLINE
void logCompilationError(io::ostream& out, ShaderStage shaderType,
        const char* name, GLuint shaderId,
        UTILS_UNUSED_IN_RELEASE CString const& sourceCode) noexcept {

    auto to_string = [](ShaderStage type) -> const char* {
        switch (type) {
            case ShaderStage::VERTEX:   return "vertex";
            case ShaderStage::FRAGMENT: return "fragment";
            case ShaderStage::COMPUTE:  return "compute";
        }
    };

    { // scope for the temporary string storage
        GLint length = 0;
        glGetShaderiv(shaderId, GL_INFO_LOG_LENGTH, &length);

        CString infoLog(length);
        glGetShaderInfoLog(shaderId, length, nullptr, infoLog.data());

        out << "Compilation error in " << to_string(shaderType) << " shader \"" << name << "\":\n"
            << "\"" << infoLog.c_str() << "\""
            << io::endl;
    }

#ifndef NDEBUG
    std::string_view const shader{ sourceCode.data(), sourceCode.size() };
    size_t lc = 1;
    size_t start = 0;
    std::string line;
    while (true) {
        size_t const end = shader.find('\n', start);
        if (end == std::string::npos) {
            line = shader.substr(start);
        } else {
            line = shader.substr(start, end - start);
        }
        out << lc++ << ":   " << line.c_str() << '\n';
        if (end == std::string::npos) {
            break;
        }
        start = end + 1;
    }
    out << io::endl;
#endif
}

UTILS_NOINLINE
void logProgramLinkError(io::ostream& out, char const* name, GLuint program) noexcept {
    GLint length = 0;
    glGetProgramiv(program, GL_INFO_LOG_LENGTH, &length);

    CString infoLog(length);
    glGetProgramInfoLog(program, length, nullptr, infoLog.data());

    out << "Link error in \"" << name << "\":\n"
        << "\"" << infoLog.c_str() << "\""
        << io::endl;
}


} // namespace filament::backend
