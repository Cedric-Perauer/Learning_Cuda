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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/cedric/torch_test

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/cedric/torch_test/build

# Include any dependencies generated for this target.
include CMakeFiles/example-app.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/example-app.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/example-app.dir/flags.make

CMakeFiles/example-app.dir/example.cpp.o: CMakeFiles/example-app.dir/flags.make
CMakeFiles/example-app.dir/example.cpp.o: ../example.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/cedric/torch_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/example-app.dir/example.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/example-app.dir/example.cpp.o -c /home/cedric/torch_test/example.cpp

CMakeFiles/example-app.dir/example.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/example-app.dir/example.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/cedric/torch_test/example.cpp > CMakeFiles/example-app.dir/example.cpp.i

CMakeFiles/example-app.dir/example.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/example-app.dir/example.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/cedric/torch_test/example.cpp -o CMakeFiles/example-app.dir/example.cpp.s

# Object files for target example-app
example__app_OBJECTS = \
"CMakeFiles/example-app.dir/example.cpp.o"

# External object files for target example-app
example__app_EXTERNAL_OBJECTS =

example-app: CMakeFiles/example-app.dir/example.cpp.o
example-app: CMakeFiles/example-app.dir/build.make
example-app: /home/cedric/libtorch/lib/libtorch.so
example-app: /home/cedric/libtorch/lib/libc10.so
example-app: /usr/local/cuda-10.2/lib64/stubs/libcuda.so
example-app: /usr/local/cuda-10.2/lib64/libnvrtc.so
example-app: /usr/local/cuda-10.2/lib64/libnvToolsExt.so
example-app: /usr/local/cuda-10.2/lib64/libcudart.so
example-app: /home/cedric/libtorch/lib/libc10_cuda.so
example-app: /home/cedric/libtorch/lib/libc10_cuda.so
example-app: /home/cedric/libtorch/lib/libc10.so
example-app: /usr/local/cuda-10.2/lib64/libcufft.so
example-app: /usr/local/cuda-10.2/lib64/libcurand.so
example-app: /usr/lib/x86_64-linux-gnu/libcublas.so
example-app: /usr/local/cuda-10.2/lib64/libcudnn.so
example-app: /usr/local/cuda-10.2/lib64/libnvToolsExt.so
example-app: /usr/local/cuda-10.2/lib64/libcudart.so
example-app: CMakeFiles/example-app.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/cedric/torch_test/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable example-app"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/example-app.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/example-app.dir/build: example-app

.PHONY : CMakeFiles/example-app.dir/build

CMakeFiles/example-app.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/example-app.dir/cmake_clean.cmake
.PHONY : CMakeFiles/example-app.dir/clean

CMakeFiles/example-app.dir/depend:
	cd /home/cedric/torch_test/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/cedric/torch_test /home/cedric/torch_test /home/cedric/torch_test/build /home/cedric/torch_test/build /home/cedric/torch_test/build/CMakeFiles/example-app.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/example-app.dir/depend

