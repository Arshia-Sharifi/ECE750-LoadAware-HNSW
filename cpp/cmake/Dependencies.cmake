# Dependencies (IMPLEMENTATION_PLAN.md §2, §6 P0.3). The C++ code needs no HDF5: datasets are .npy files.
# Offline builds: pass -DFETCHCONTENT_SOURCE_DIR_<NAME>=<dir> for each dependency (NAME in upper case).
include(FetchContent)

if(LAEF_BUILD_TESTS)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  FetchContent_Declare(googletest GIT_REPOSITORY https://github.com/google/googletest.git GIT_TAG v1.14.0)
  FetchContent_MakeAvailable(googletest)
endif()

# nlohmann/json: run configs, dataset metadata, price list, profile and JSON outputs.
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
FetchContent_Declare(nlohmann_json URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
FetchContent_MakeAvailable(nlohmann_json)

# CLI11: command lines of the apps.
set(CLI11_TESTING OFF CACHE INTERNAL "")
FetchContent_Declare(CLI11 GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git GIT_TAG v2.4.2)
FetchContent_MakeAvailable(CLI11)

if(LAEF_WITH_ADAEF)
  # Eigen and Boost may be system packages or header-only checkouts:
  #   -DLAEF_EIGEN_INCLUDE=/path/to/eigen   or a system Eigen3 3.4 (find_package)
  #   -DLAEF_BOOST_INCLUDE=/path/to/boost_math/include (Boost.Math standalone is enough for Ada-ef)
  set(LAEF_EIGEN_INCLUDE "" CACHE PATH "Header-only Eigen checkout (optional)")
  set(LAEF_BOOST_INCLUDE "" CACHE PATH "Header-only Boost(.Math) include directory (optional)")
  if(LAEF_EIGEN_INCLUDE)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES INTERFACE_INCLUDE_DIRECTORIES ${LAEF_EIGEN_INCLUDE})
  else()
    find_package(Eigen3 3.4 REQUIRED NO_MODULE)
  endif()
  if(LAEF_BOOST_INCLUDE)
    add_library(Boost::headers INTERFACE IMPORTED)
    set_target_properties(Boost::headers PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES ${LAEF_BOOST_INCLUDE}
      INTERFACE_COMPILE_DEFINITIONS BOOST_MATH_STANDALONE)
  else()
    find_package(Boost REQUIRED)
  endif()
  find_package(OpenMP REQUIRED)
endif()
