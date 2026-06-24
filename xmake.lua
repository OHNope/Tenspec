set_project("typetorch")
set_version("0.1.0")
set_languages("c++26")
set_policy("build.c++.modules.gcc.cxx11abi", true)

local local_config = {}

function typetorch_config(config)
    local_config = config or {}
end

if os.isfile("typetorch.local.lua") then
    includes("typetorch.local.lua")
end

local function config_value(name, env_name)
    local env_value = os.getenv(env_name or name)
    if env_value and env_value ~= "" then
        return env_value
    end
    local value = local_config[name]
    if value ~= nil and value ~= "" then
        return value
    end
    return nil
end

local function detect_gcc_relocate_ldflags()
    local configured_flags = config_value("gcc_relocate_ldflags", "TYPETORCH_GCC_RELOCATE_LDFLAGS")
    if configured_flags then return configured_flags end

    local candidates = {}
    local gcc_root = config_value("gcc_root", "GCC_ROOT")
    local configured_toolchain = config_value("gcc_toolchain_root", "GCC_TOOLCHAIN_ROOT")
    if configured_toolchain then
        table.insert(candidates, configured_toolchain)
    end
    if gcc_root and gcc_root ~= "" then
        local triplet = config_value("gcc_triplet", "GCC_TRIPLET") or "x86_64-focal-linux-gnu"
        table.insert(candidates, path.join(path.directory(gcc_root), "x-tools", triplet))
    end

    for _, root in ipairs(candidates) do
        local triplet = config_value("gcc_triplet", "GCC_TRIPLET") or "x86_64-focal-linux-gnu"
        local version = config_value("gcc_version", "GCC_VERSION") or "16.1.0"
        local frontend_dir = path.join(root, "libexec", "gcc", triplet, version)
        local gcc_lib_dir = path.join(root, "lib", "gcc", triplet, version)
        local sysroot = config_value("gcc_sysroot", "GCC_SYSROOT") or path.join(root, triplet, "sysroot")
        if os.isfile(path.join(frontend_dir, "cc1plus")) and
           os.isfile(path.join(gcc_lib_dir, "crtbeginS.o")) and
           os.isfile(path.join(sysroot, "usr", "lib", "crti.o")) then
            return format("-B%s/ -B%s/ -B%s/ -B%s/ --sysroot=%s",
                          frontend_dir,
                          gcc_lib_dir,
                          path.join(sysroot, "usr", "lib"),
                          path.join(sysroot, "lib"),
                          sysroot)
        end
    end
    return nil
end

local gcc_relocate_ldflags = detect_gcc_relocate_ldflags()
if gcc_relocate_ldflags and gcc_relocate_ldflags ~= "" then
    for flag in gcc_relocate_ldflags:gmatch("%S+") do
        add_ldflags(flag, {force = true})
        add_shflags(flag, {force = true})
    end
end

add_rules("mode.debug", "mode.release")
add_repositories("local-repo .")
add_cxxflags("-fmodules", "-freflection", "-Wall"," -Wextra","-fvisibility=hidden", "-ffunction-sections", "-fdata-sections", {force = true})
    add_ldflags("-Wl,--gc-sections", {force = true})
    add_shflags("-Wl,--gc-sections", {force = true})

local configured_python_include = config_value("python_include_dir", "PYTHON_INCLUDE_DIR")
if configured_python_include then
    add_cxxflags("-isystem" .. configured_python_include, {force = true})
    add_sysincludedirs(configured_python_include)
end

local libtorch_variant = os.getenv("TYPETORCH_LIBTORCH_VARIANT") or "cu128"
local libtorch_version = libtorch_variant == "cpu" and "2.8.0+cpu" or "2.8.0+cu128"
add_requires("libtorch-bin " .. libtorch_version, {alias = "libtorch"})

local typetorch_module_files = {
    "src/libtorch.mpp",
    "src/typetorch/types.mpp",
    "src/typetorch/torch_mappings.mpp",
    "src/typetorch/shape_meta.mpp",
    "src/typetorch/common.mpp",
    "src/typetorch/runtime_checks.mpp",
    "src/typetorch/tensor/core.mpp",
    "src/typetorch/tensor/factory.mpp",
    "src/typetorch/tensor/arithmetic.mpp",
    "src/typetorch/tensor/view.mpp",
    "src/typetorch/tensor/nn.mpp",
    "src/typetorch/tensor/tensor.mpp",
    "src/typetorch/nnModules/core.mpp",
    "src/typetorch/nnModules/linear.mpp",
    "src/typetorch/nnModules/conv2d.mpp",
    "src/typetorch/nnModules/embedding.mpp",
    "src/typetorch/nnModules/layer_norm.mpp",
    "src/typetorch/nnModules/rms_norm.mpp",
    "src/typetorch/nnModules/nnModules.mpp",
    "src/typetorch/typetorch.mpp"
}

local typetorch_examples_module_files = {
    "src/examples/typetorch_examples.mpp",
    "src/examples/examples.cpp",
}

local function add_typetorch_modules()
    for _, file in ipairs(typetorch_module_files) do
        add_files(file)
    end
end

local function add_typetorch_examples_modules()
    for _, file in ipairs(typetorch_examples_module_files) do
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
        if not torch then
            return
        end
        local torchdir = torch:installdir()
        target:add("defines", format("TYPETORCH_LIBTORCH_ROOT=\"%s\"", torch:installdir()))
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

local debug_cxxflags = {"-O0", "-g3", "-fno-inline", "-fno-omit-frame-pointer"}
local release_cxxflags = {"-O2", "-g1", "-fno-omit-frame-pointer"}

local function add_each(add, values)
    for _, value in ipairs(values or {}) do
        add(value)
    end
end

local function add_mode_flags()
    if is_mode("debug") then
        add_each(add_cxxflags, debug_cxxflags)
        add_defines("TYPETORCH_DEBUG")
        if get_config("sanitizers") then
            add_cxxflags("-fsanitize=address,undefined")
            add_ldflags("-fsanitize=address,undefined")
        end
    elseif is_mode("release") then
        add_each(add_cxxflags, release_cxxflags)
        add_defines("TYPETORCH_RELEASE")
    end
end

local function add_fast_io()
    add_files("src/fast_io.mpp", "src/fast_io.cpp")
    add_includedirs("third_party/fast_io/include")
end

local function add_target_files(files)
    for _, file in ipairs(files or {}) do
        add_files(file)
    end
end

local function configure_libtorch_target(config)
    config = config or {}
    if config.default ~= nil then
        set_default(config.default)
    end
    set_kind(config.kind or "binary")
    add_rules("libtorch_runtime", config.python_rule or "python_headers")
    add_packages("libtorch")
    add_includedirs("src")
    if config.torch_python then
        add_links("torch_python")
    end
    if config.root_includedir then
        add_includedirs(".")
    end
    if config.libtorch_module_only then
        add_files("src/libtorch.mpp")
    else
        add_typetorch_modules()
    end
    if config.fast_io ~= false then
        add_fast_io()
    end
    if config.examples then
        add_typetorch_examples_modules()
    end
    add_target_files(config.files)
    add_target_files(config.extra_files)
    add_mode_flags()
end

target("typetorch_cpp_debug")
    configure_libtorch_target({
        examples = true,
        files = {"examples/cpp_debug.cpp"},
    })
target_end()

target("typetorch_forwarding_benchmark")
    configure_libtorch_target({
        default = false,
        files = {"benchmarks/forwarding_benchmark.cpp"},
    })
target_end()

target("binary_size_tensor_probe")
    configure_libtorch_target({
        default = false,
        files = {"tests/binary_size_tensor_probe.cpp"},
    })
target_end()

target("binary_size_tensor_checked_probe")
    configure_libtorch_target({
        default = false,
        files = {"tests/binary_size_tensor_checked_probe.cpp"},
    })
target_end()

target("binary_size_tensor_unsafe_probe")
    configure_libtorch_target({
        default = false,
        files = {"tests/binary_size_tensor_unsafe_probe.cpp"},
    })
target_end()

target("binary_size_libtorch_probe")
    configure_libtorch_target({
        default = false,
        libtorch_module_only = true,
        files = {"tests/binary_size_libtorch_probe.cpp"},
    })
target_end()

target("typetorch_tensor_arithmetic_test")
    configure_libtorch_target({
        default = false,
        files = {"tests/tensor_arithmetic_test.cpp"},
    })
target_end()

target("typetorch_sdk")
    configure_libtorch_target({
        kind = "static",
        default = false,
    })
target_end()

target("typetorch_capi_ext")
    configure_libtorch_target({
        default = false,
        kind = "shared",
        python_rule = "python_extension",
        torch_python = true,
        root_includedir = true,
        examples = true,
        files = {
            "src/bindings/python.mpp",
            "src/bindings/torch_python.mpp",
            "src/bindings/torch_python.cpp",
            "src/bindings/capi_bridge.mpp",
            "src/bindings/capi_python_cast.mpp",
            "src/bindings/capi_invoke.mpp",
            "src/bindings/capi_reflect_namespace.mpp",
            "src/bindings/capi_module.cpp",
            "src/bindings/capi_reflect.mpp",
        },
    })
    set_prefixname("")
    set_basename("typetorch_capi")
target_end()
