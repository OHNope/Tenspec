set_project("tenspec")
set_version("0.1.0")
set_languages("c++26")
set_policy("build.c++.modules.gcc.cxx11abi", true)

add_rules("mode.debug", "mode.release")
add_repositories("local-repo .")
add_cxxflags("-fmodules", "-freflection", "-Wall"," -Wextra","-fvisibility=hidden", {force = true})
add_cxxflags("-isystem/usr/include/python3.9", {force = true})

add_requires("libtorch-bin 2.8.0+cu128", {alias = "libtorch"})

local tenspec_module_files = {
    "src/libtorch.mpp",
    "src/tenspec_types.mpp",
    "src/tenspec_aten_mappings.mpp",
    "src/tenspec_shape_meta.mpp",
    "src/tenspec_common.mpp",
    "src/tenspec_runtime_checks.mpp",
    "src/tenspec_tensor.mpp",
    "src/tenspec.mpp"
}

local tenspec_examples_module_files = {
    "src/tenspec_examples.mpp",
    "src/examples.cpp",
}

local function add_tenspec_modules()
    for _, file in ipairs(tenspec_module_files) do
        add_files(file)
    end
end

local function add_tenspec_examples_modules()
    for _, file in ipairs(tenspec_examples_module_files) do
        add_files(file)
    end
end

local function default_python()
    local python_env = os.getenv("PYTHON")
    if python_env then
        return python_env
    end
    for _, candidate in ipairs({".venv/bin/python", "venv/bin/python"}) do
        if os.isfile(candidate) then
            return candidate
        end
    end
    return "python3"
end

option("python")
    set_default(default_python())
    set_showmenu(true)
    set_description("Python interpreter used for extension metadata")
option_end()

option("sanitizers")
    set_default(false)
    set_showmenu(true)
    set_description("Enable AddressSanitizer and UndefinedBehaviorSanitizer for debug builds")
option_end()

local python = get_config("python")

local function add_python_sysincludedirs(target)
    local seen = {}
    local dirs = {
        os.getenv("PYTHON_INCLUDE_DIR"),
        ".venv/include",
        ".venv/include/python3.9",
        ".venv/include/python3.10",
        ".venv/include/python3.11",
        ".venv/include/python3.12",
        "venv/include",
        "venv/include/python3.9",
        "venv/include/python3.10",
        "venv/include/python3.11",
        "venv/include/python3.12",
        "/usr/include/python3.9",
        "/usr/include/python3.10",
        "/usr/include/python3.11",
        "/usr/include/python3.12",
        "/usr/local/include/python3.9",
        "/usr/local/include/python3.10",
        "/usr/local/include/python3.11",
        "/usr/local/include/python3.12",
    }
    for _, dir in ipairs(dirs) do
        if dir and dir ~= "" and not seen[dir] then
            target:add("sysincludedirs", dir)
            seen[dir] = true
        end
    end
end

rule("libtorch_runtime")
    on_load(function (target)
        local torch = target:pkg("libtorch")
        local torchdir = torch:installdir()
        target:add("defines", format("TENSPEC_LIBTORCH_ROOT=\"%s\"", torch:installdir()))
        target:add("sysincludedirs", path.join(torchdir, "include"))
        target:add("sysincludedirs", path.join(torchdir, "include", "torch", "csrc", "api", "include"))
        target:add("linkdirs", path.join(torchdir, "lib"))
        target:add("rpathdirs", path.join(torchdir, "lib"))
        target:add("links", "torch", "torch_cpu", "c10")
    end)
rule_end()

rule("python_extension")
    on_load(function (target)
        add_python_sysincludedirs(target)
    end)
    after_load(function (target)
        target:set("extension", ".so")
    end)
rule_end()

rule("python_headers")
    on_load(function (target)
        add_python_sysincludedirs(target)
    end)
rule_end()

target("tenspec_cpp_debug")
    set_kind("binary")
    add_rules("libtorch_runtime", "python_headers")
    add_packages("libtorch")
    add_files("src/fastio.mpp", "src/fastio.cpp", "examples/cpp_debug.cpp")
    add_tenspec_modules()
    add_tenspec_examples_modules()
    add_includedirs("third_party/fast_io/include")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
        add_defines("TENSPEC_DEBUG")
        if get_config("sanitizers") then
            add_cxxflags("-fsanitize=address,undefined")
            add_ldflags("-fsanitize=address,undefined")
        end
    elseif is_mode("release") then
        add_cxxflags("-O3", "-march=native", "-flto")
        add_ldflags("-flto")
        add_defines("TENSPEC_RELEASE")
    end
target_end()

target("tenspec_forwarding_benchmark")
    set_default(false)
    set_kind("binary")
    add_rules("libtorch_runtime", "python_headers")
    add_packages("libtorch")
    add_files("src/fastio.mpp", "src/fastio.cpp", "benchmarks/forwarding_benchmark.cpp")
    add_tenspec_modules()
    add_includedirs("third_party/fast_io/include")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
        add_defines("TENSPEC_DEBUG")
    elseif is_mode("release") then
        add_cxxflags("-O3", "-march=native", "-flto")
        add_ldflags("-flto")
        add_defines("TENSPEC_RELEASE")
    end
target_end()

target("binary_size_tensor_probe")
    set_default(false)
    set_kind("binary")
    add_rules("libtorch_runtime", "python_headers")
    add_packages("libtorch")
    add_files("src/fastio.mpp", "src/fastio.cpp", "tests/binary_size_tensor_probe.cpp")
    add_tenspec_modules()
    add_includedirs("third_party/fast_io/include")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
    elseif is_mode("release") then
        add_cxxflags("-O2", "-g1", "-fno-omit-frame-pointer")
    end
target_end()

target("binary_size_libtorch_probe")
    set_default(false)
    set_kind("binary")
    add_rules("libtorch_runtime", "python_headers")
    add_packages("libtorch")
    add_files("src/libtorch.mpp")
    add_files("src/fastio.mpp", "src/fastio.cpp", "tests/binary_size_libtorch_probe.cpp")
    add_includedirs("third_party/fast_io/include")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
    elseif is_mode("release") then
        add_cxxflags("-O2", "-g1", "-fno-omit-frame-pointer")
    end
target_end()

target("tenspec_tensor_arithmetic_test")
    set_default(false)
    set_kind("binary")
    add_rules("libtorch_runtime", "python_headers")
    add_packages("libtorch")
    add_files("src/fastio.mpp", "src/fastio.cpp", "tests/tensor_arithmetic_test.cpp")
    add_tenspec_modules()
    add_includedirs("third_party/fast_io/include")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
    elseif is_mode("release") then
        add_cxxflags("-O2", "-g1", "-fno-omit-frame-pointer")
    end
target_end()

target("core")
    set_kind("static")
    add_rules("libtorch_runtime", "python_headers")
    add_packages("libtorch")
    add_links("torch_python")
    add_includedirs("third_party/fast_io/include")
    add_files("src/fastio.mpp", "src/fastio.cpp", "src/test.cpp")
    add_tenspec_modules()
    add_cxxflags("-fvisibility=hidden")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
        add_defines("TENSPEC_DEBUG")
        if get_config("sanitizers") then
            add_cxxflags("-fsanitize=address,undefined")
            add_ldflags("-fsanitize=address,undefined")
        end
    elseif is_mode("release") then
        add_cxxflags("-O3", "-march=native", "-flto")
        add_ldflags("-flto")
        add_defines("TENSPEC_RELEASE")
    end
target_end()

target("tenspec_capi_ext")
    set_default(false)
    set_kind("shared")
    add_rules("libtorch_runtime", "python_extension")
    add_packages("libtorch")
    add_links("torch_python")
    add_files("bindings/python.mpp")
    add_files("bindings/capi_bridge.mpp")
    add_files("bindings/capi_module.cpp")
    add_files("src/tenspec_capi_reflect.mpp")
    add_tenspec_modules()
    add_tenspec_examples_modules()
    add_includedirs(".", "third_party/fast_io/include")
    add_cxxflags("-fvisibility=hidden")
    if is_mode("debug") then
        add_cxxflags("-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer")
        add_defines("TENSPEC_DEBUG")
    elseif is_mode("release") then
        add_cxxflags("-fno-unwind-tables", "-fno-asynchronous-unwind-tables")
        add_cxxflags("-O3", "-march=native", "-flto")
        add_ldflags("-flto")
        add_defines("TENSPEC_RELEASE")
    end
    set_prefixname("")
    set_basename("tenspec_capi")
target_end()
