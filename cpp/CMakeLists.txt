# LA-ef C++ build (IMPLEMENTATION_PLAN.md §6 P0.3).
# Single-TU rule (§0.3): only src/adaef_backend.cpp may include hnswlib/*.h, because the Ada-ef headers
# define non-inline functions. Everything else talks to it through laef::AnnBackend.
#
#   cmake -S cpp -B build                         # everything (Eigen3, Boost, OpenMP, FetchContent)
#   cmake -S cpp -B build -DLAEF_WITH_ADAEF=OFF   # core, fake backend and the header tests (no Ada-ef)
# Binaries are written to build/bin.
cmake_minimum_required(VERSION 3.22)
project(laef LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(LAEF_NATIVE "Compile with -march=native" ON)
option(LAEF_WITH_ADAEF "Build the Ada-ef backend, the apps and the backend parity test" ON)
option(LAEF_BUILD_TESTS "Build the unit tests" ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
if(LAEF_NATIVE)
  add_compile_options(-march=native)
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(cmake/Dependencies.cmake)
find_package(Threads REQUIRED)

# Header-only online core (controllers, bucket, sensor window, price list, allocators, planner).
add_library(laef_headers INTERFACE)
target_include_directories(laef_headers INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(laef_headers INTERFACE Threads::Threads nlohmann_json::nlohmann_json)

# Core library: everything in src/ except the Ada-ef backend and the fake backend.
add_library(laef_core STATIC
  src/dataset_io.cpp
  src/workload.cpp
  src/price_list.cpp
  src/calibration.cpp
  src/engine.cpp
  src/results_writer.cpp
)
target_link_libraries(laef_core PUBLIC laef_headers)

# Deterministic backend for engine tests (no index needed).
add_library(laef_fake STATIC src/fake_backend.cpp)
target_link_libraries(laef_fake PUBLIC laef_headers)

if(LAEF_WITH_ADAEF)
  set(ADAEF_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/hnsw-ada-ef CACHE PATH "hnsw-ada-ef checkout (contains hnswlib/)")
  add_library(laef_adaef STATIC src/adaef_backend.cpp)
  target_include_directories(laef_adaef PRIVATE ${ADAEF_DIR})
  target_link_libraries(laef_adaef PUBLIC laef_headers PRIVATE Eigen3::Eigen Boost::headers OpenMP::OpenMP_CXX)

  foreach(app laef_build_index laef_calibrate laef_profile laef_bench)
    add_executable(${app} apps/${app}.cpp)
    target_link_libraries(${app} PRIVATE laef_core laef_adaef CLI11::CLI11)
  endforeach()
endif()

if(LAEF_BUILD_TESTS)
  enable_testing()
  include(GoogleTest)

  # Header-level tests. test_formats.cpp was merged into test_backend_parity.cpp (plan §13.1).
  foreach(t test_p_aimd test_token_bucket test_price_list test_fixed_aimd test_sensor_window test_allocators)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/${t}.cpp)
      add_executable(${t} tests/${t}.cpp)
      target_link_libraries(${t} PRIVATE laef_headers GTest::gtest_main)
      gtest_discover_tests(${t})
    endif()
  endforeach()

  # Engine QPS matrix (plan §13.3). Needs the core (engine, workload, results writer) and the fake backend.
  # Real-dataset cases skip at run time, so this target links no Ada-ef code.
  add_executable(test_engine_qps tests/test_engine_qps.cpp)
  target_link_libraries(test_engine_qps PRIVATE laef_core laef_fake laef_headers GTest::gtest_main Threads::Threads)
  target_compile_definitions(test_engine_qps PRIVATE LAEF_REPO_ROOT="${CMAKE_CURRENT_SOURCE_DIR}/..")
  # Timing-sensitive engine tests: RUN_SERIAL keeps other tests from competing for CPU while they run.
gtest_discover_tests(test_engine_qps DISCOVERY_TIMEOUT 120 PROPERTIES RUN_SERIAL TRUE)

  if(LAEF_WITH_ADAEF)
    # Unity TU: includes src/adaef_backend.cpp directly, so it must NOT link laef_adaef.
    add_executable(test_backend_parity tests/test_backend_parity.cpp)
    target_include_directories(test_backend_parity PRIVATE ${ADAEF_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_definitions(test_backend_parity PRIVATE LAEF_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
    target_link_libraries(test_backend_parity PRIVATE Eigen3::Eigen Boost::headers OpenMP::OpenMP_CXX Threads::Threads GTest::gtest_main)
    gtest_discover_tests(test_backend_parity DISCOVERY_TIMEOUT 60)
  endif()
endif()
