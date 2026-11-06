# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import re
import shutil
import sys
import subprocess
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from setuptools import find_packages

MIN_CMAKE = (3, 20)

# Supported installation: build the MLIR Python bindings from source against an
# LLVM/MLIR build named by LLVM_BUILD_DIR (docs/source/setup/index.rst,
# docs/release/INSTALL.md). CMake writes the bindings to
# mlir/build/tools/allo/_mlir. Editable installs reach them through the tracked
# symlink allo/_mlir; every other build copies them into the wheel.
MLIR_BINDINGS_SUBDIR = os.path.join("tools", "allo", "_mlir")

# Non-Python files the package reads at runtime through __file__-relative
# paths. tests/perf/test_packaging.py checks that the sdist carries them.
PACKAGE_DATA = {
    "allo": [
        "ir/template/*.mlir",
        "library/aie/kernels/*.cc",
        # namespace package (no __init__.py), imported as allo.library.aie.modules
        "library/aie/modules/*.py",
        "backend/aie/README.md",
    ],
    "allo.autoscheduler": ["outline_loops.mlir"],
    "allo.harness": [
        "aie/*",
        "catapult/*",
        "pynq/*",
        "tapa/*",
        "vitis/*",
        "vivado/*",
        "xlscc/*",
    ],
    "allo.harness.makefile_gen": ["description.json"],
    "allo.pim": ["templates/apu_g2/*", "templates/apu_g2/device/*"],
}


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        # Ensure CMake is installed
        try:
            version_text = subprocess.check_output(["cmake", "--version"], text=True)
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )
        found = re.search(r"version (\d+)\.(\d+)", version_text)
        if found and tuple(int(v) for v in found.groups()) < MIN_CMAKE:
            raise RuntimeError(
                f"CMake >= {MIN_CMAKE[0]}.{MIN_CMAKE[1]} is required to build the "
                "Allo MLIR bindings, but `cmake` on PATH is "
                f"{found.group(1)}.{found.group(2)}. Run "
                "`python3 -m pip install cmake` and put its bin directory first "
                "on PATH."
            )

        # Call the build process for each extension
        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        # Retrieve LLVM_BUILD_DIR from environment variable
        llvm_build_dir = os.environ.get("LLVM_BUILD_DIR")
        if not llvm_build_dir:
            raise RuntimeError(
                "LLVM_BUILD_DIR environment variable is not set. Allo's MLIR "
                "Python bindings are built from source against an LLVM/MLIR "
                "build; see docs/source/setup/index.rst (Install from Source) "
                "and docs/release/INSTALL.md."
            )
        mlir_config = os.path.join(
            llvm_build_dir, "lib", "cmake", "mlir", "MLIRConfig.cmake"
        )
        if not os.path.isfile(mlir_config):
            raise RuntimeError(
                f"LLVM_BUILD_DIR={llvm_build_dir} is not an MLIR build: "
                f"{mlir_config} does not exist. Build LLVM with "
                "-DLLVM_ENABLE_PROJECTS=mlir and -DMLIR_ENABLE_BINDINGS_PYTHON=ON "
                "(docs/source/setup/index.rst)."
            )

        # Auto-detect nanobind cmake directory if not set
        nanobind_cmake_dir = os.environ.get("NANOBIND_CMAKE_DIR")
        if not nanobind_cmake_dir:
            try:
                import nanobind

                nanobind_cmake_dir = nanobind.cmake_dir()
            except ImportError:
                raise RuntimeError(
                    "nanobind is not installed. Please run:\n\n"
                    "  python3 -m pip install nanobind"
                )
        cmake_args = [
            f"-DMLIR_DIR={llvm_build_dir}/lib/cmake/mlir",
            f"-DPython3_EXECUTABLE={sys.executable}",
            # nanobind runs find_package(Python); without this hint it can pick
            # another interpreter on PATH and emit the wrong ABI suffix.
            f"-DPython_EXECUTABLE={sys.executable}",
            # $ORIGIN-relative build RPATHs keep the copied bindings loadable
            # after the build tree is removed.
            "-DCMAKE_BUILD_RPATH_USE_ORIGIN=ON",
            f"-Dnanobind_DIR={nanobind_cmake_dir}",
            "-DMLIR_BINDINGS_PYTHON_NB_DOMAIN=allo",
        ]

        build_temp = os.path.join(ext.sourcedir, "build")
        if not os.path.exists(build_temp):
            os.makedirs(build_temp)

        BUILD_WITH = os.environ.get("BUILD_WITH")
        if not BUILD_WITH or BUILD_WITH == "ninja":
            subprocess.run(
                ["cmake", "-G Ninja", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(
                    ["ninja", f"-j{NUM_THREADS}"], cwd=build_temp, check=True
                )
            else:
                subprocess.run(["ninja"], cwd=build_temp, check=True)
        elif BUILD_WITH == "make":
            subprocess.run(
                ["cmake", "-G Unix Makefiles", ext.sourcedir] + cmake_args,
                cwd=build_temp,
                check=True,
            )
            if NUM_THREADS := os.environ.get("NUM_THREADS"):
                subprocess.run(["make", f"-j{NUM_THREADS}"], cwd=build_temp, check=True)
            else:
                subprocess.run(["make", "-j"], cwd=build_temp, check=True)
        else:
            raise RuntimeError(f"Unsupported BUILD_WITH={BUILD_WITH}")
        self.install_bindings(build_temp)

    def install_bindings(self, build_temp):
        built = os.path.join(build_temp, MLIR_BINDINGS_SUBDIR)
        if not os.path.isdir(built):
            raise RuntimeError(
                f"the CMake build finished without producing {built}; "
                "the Allo MLIR Python bindings are missing"
            )
        if self.inplace or getattr(self, "editable_mode", False):
            return
        # Wheels cannot hold symlinks. Links into the LLVM source tree are
        # copied as files; in-tree aliases (libX.so -> libX.so.<ver>) are
        # dropped because the extensions load the versioned soname.
        root = os.path.realpath(built) + os.sep

        def ignore(directory, names):
            skipped = set()
            for name in names:
                path = os.path.join(directory, name)
                if name == "__pycache__" or name.endswith(".pyc"):
                    skipped.add(name)
                elif os.path.islink(path) and os.path.realpath(path).startswith(root):
                    skipped.add(name)
            return skipped

        dest = os.path.join(self.build_lib, "allo", "_mlir")
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(built, dest, ignore=ignore)


def parse_requirements(filename):
    """Load requirements from a pip requirements file.

    Only the runtime list (requirements.txt) is read; development, test and
    optional-feature tools live in requirements-dev.txt, which pip options
    such as ``-r`` or direct URLs make unsuitable for install_requires.
    """
    lineiter = (line.strip() for line in open(filename))
    requirements = [line for line in lineiter if line and not line.startswith("#")]
    for line in requirements:
        if line.startswith("-") or " @ " in line:
            raise RuntimeError(
                f"{filename}: '{line}' is not a plain requirement; keep pip "
                "options and direct URLs in requirements-dev.txt"
            )
    return requirements


if __name__ == "__main__":
    with open("README.md", encoding="utf-8") as fp:
        long_description = fp.read()

    setup(
        name="allo",
        description="Allo",
        version="0.5",
        author="Allo Community",
        long_description=long_description,
        long_description_content_type="text/markdown",
        setup_requires=["nanobind>=2.9,<3.0"],
        install_requires=parse_requirements("requirements.txt"),
        # examples/, benchmarks/ and tests/ are repository trees, not part of
        # the installed package; allo._mlir is installed by build_ext.
        packages=find_packages(
            include=["allo", "allo.*"], exclude=["allo._mlir", "allo._mlir.*"]
        ),
        package_data=PACKAGE_DATA,
        ext_modules=[CMakeExtension("mlir", sourcedir="mlir")],
        cmdclass={"build_ext": CMakeBuild},
        url="https://github.com/cornell-zhang/allo",
        # Allo's license plus the texts of the third parties the distributions
        # carry (THIRD_PARTY_NOTICES.md); setuptools installs them under
        # dist-info/licenses/ and the sdist includes them through MANIFEST.in.
        license="Apache-2.0",
        license_files=[
            "LICENSE",
            "THIRD_PARTY_NOTICES.md",
            "THIRD_PARTY_LICENSES/*",
        ],
        python_requires=">=3.12",
        classifiers=[
            "Programming Language :: Python :: 3.12",
            "Topic :: Scientific/Engineering",
            "Topic :: System :: Hardware",
            "Operating System :: OS Independent",
        ],
        zip_safe=True,
    )
