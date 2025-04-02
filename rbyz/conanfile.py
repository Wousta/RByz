from conan import ConanFile
from conan.tools.cmake import CMake


class ex(ConanFile):
    name = "ex"
    version = "1.0"
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeToolchain", "CMakeDeps"  # Use the new toolchain system

    def requirements(self):
        self.requires("lyra/[*]")

    def build(self):
        cmake = CMake(self)
        cmake.configure()
        cmake.build()

