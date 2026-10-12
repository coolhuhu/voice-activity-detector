import re
import os
import platform
import shutil
from pathlib import Path

import setuptools
from setuptools.command.build_ext import build_ext


package_name = "voice_activity_detector"

def is_windows():
    return platform.system() == "Windows"


def is_macos():
    return platform.system() == "Darwin"


def is_linux():
    return platform.system() == "Linux"


def is_arm64():
    return platform.machine() in ["arm64", "aarch64"]


def is_x86():
    return platform.machine() in ["i386", "i686", "x86_64"]


def get_package_version() -> str:
    with open("CMakeLists.txt", "r") as f:
        content = f.read()
        
    match = re.search(
        r"set\(\s*VOICE_ACTIVITY_DETECTOR_VERSION\s+\"?([^\"\s)]+)\"?\s*\)",
        content,
    )
    if not match:
        raise RuntimeError("VOICE_ACTIVITY_DETECTOR_VERSION not found in CMakeLists.txt")

    return match.group(1)


def cmake_extension(name: str, *args, **kwargs) -> setuptools.Extension:
    kwargs["language"] = "c++"
    sources = []
    return setuptools.Extension(name, sources, *args, **kwargs)


class BuildExtension(build_ext):
    def build_extension(self, ext):
        # build/temp.linux-x86_64-cpython-<python_version>
        # such as: build/lib.linux-x86_64-cpython-313
        print(f"build/temp: {self.build_temp}")
        os.makedirs(self.build_temp, exist_ok=True)
        
        # build/lib.linux-x86_64-cpython-<python_version>
        # such as: build/lib.linux-x86_64-cpython-313
        print(f"build/lib: {self.build_lib}")
        os.makedirs(self.build_lib, exist_ok=True)
        
        install_dir = Path(self.build_lib).resolve() / package_name
        project_dir = Path(__file__).parent.resolve()
        
        cmake_args = " -DENABLE_PYTHON=ON"
        cmake_args += " -DCMAKE_BUILD_TYPE=Release"
        cmake_args += " -DBUILD_SHARED_LIBS=ON"
        cmake_args += f" -DCMAKE_INSTALL_PREFIX={install_dir}"
        
        cmake_build_args = " -j 4"
        
        if is_windows() or is_macos():
            raise NotImplementedError("Windows and Macos is currently not supported")
        
        build_cmd = f"""
            cd {self.build_temp}
            cmake {cmake_args} {project_dir}
            cmake --build . {cmake_build_args}
            cmake --install .
        """
        
        print(f"build command:\n{build_cmd}")
        ret = os.system(build_cmd)
        if ret != 0:
            raise Exception("<pip install .> build failed")

        # Editable installs copy the extension back into the source package.
        # Keep the CMake-installed runtime libraries beside it so imports work.
        source_lib_dir = project_dir / "python" / package_name / "lib"
        source_lib_dir.mkdir(parents=True, exist_ok=True)
        for runtime_file in (install_dir / "lib").glob("*"):
            if runtime_file.is_file():
                shutil.copy2(runtime_file, source_lib_dir / runtime_file.name)
        

setuptools.setup(
    version=get_package_version(),
    ext_modules=[cmake_extension("voice_activity_detector.lib._voice_activity_detector")],
    package_data={package_name: ["lib/*"]},
    cmdclass={"build_ext": BuildExtension}
)