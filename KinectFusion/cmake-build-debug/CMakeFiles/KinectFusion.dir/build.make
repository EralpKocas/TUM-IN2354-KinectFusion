# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.16

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = "/Users/eralpkocas/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/201.7846.88/CLion.app/Contents/bin/cmake/mac/bin/cmake"

# The command to remove a file.
RM = "/Users/eralpkocas/Library/Application Support/JetBrains/Toolbox/apps/CLion/ch-0/201.7846.88/CLion.app/Contents/bin/cmake/mac/bin/cmake" -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion"

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug"

# Include any dependencies generated for this target.
include CMakeFiles/KinectFusion.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/KinectFusion.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/KinectFusion.dir/flags.make

CMakeFiles/KinectFusion.dir/main.cpp.o: CMakeFiles/KinectFusion.dir/flags.make
CMakeFiles/KinectFusion.dir/main.cpp.o: ../main.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/KinectFusion.dir/main.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KinectFusion.dir/main.cpp.o -c "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/main.cpp"

CMakeFiles/KinectFusion.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KinectFusion.dir/main.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/main.cpp" > CMakeFiles/KinectFusion.dir/main.cpp.i

CMakeFiles/KinectFusion.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KinectFusion.dir/main.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/main.cpp" -o CMakeFiles/KinectFusion.dir/main.cpp.s

CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.o: CMakeFiles/KinectFusion.dir/flags.make
CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.o: ../FreeImageHelper.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir="/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.o"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.o -c "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/FreeImageHelper.cpp"

CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.i"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/FreeImageHelper.cpp" > CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.i

CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.s"
	/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/FreeImageHelper.cpp" -o CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.s

# Object files for target KinectFusion
KinectFusion_OBJECTS = \
"CMakeFiles/KinectFusion.dir/main.cpp.o" \
"CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.o"

# External object files for target KinectFusion
KinectFusion_EXTERNAL_OBJECTS =

KinectFusion: CMakeFiles/KinectFusion.dir/main.cpp.o
KinectFusion: CMakeFiles/KinectFusion.dir/FreeImageHelper.cpp.o
KinectFusion: CMakeFiles/KinectFusion.dir/build.make
KinectFusion: /usr/local/lib/libceres.2.0.0.dylib
KinectFusion: /usr/local/lib/libglog.0.4.0.dylib
KinectFusion: /usr/local/lib/libgflags.2.2.2.dylib
KinectFusion: CMakeFiles/KinectFusion.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir="/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug/CMakeFiles" --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable KinectFusion"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/KinectFusion.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/KinectFusion.dir/build: KinectFusion

.PHONY : CMakeFiles/KinectFusion.dir/build

CMakeFiles/KinectFusion.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/KinectFusion.dir/cmake_clean.cmake
.PHONY : CMakeFiles/KinectFusion.dir/clean

CMakeFiles/KinectFusion.dir/depend:
	cd "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug" && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion" "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion" "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug" "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug" "/Users/eralpkocas/Documents/TUM/3D Scanning & Motion Planning/TUM-IN2354-KinectFusion/KinectFusion/cmake-build-debug/CMakeFiles/KinectFusion.dir/DependInfo.cmake" --color=$(COLOR)
.PHONY : CMakeFiles/KinectFusion.dir/depend

