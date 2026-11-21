cmake_minimum_required(VERSION 3.16.0)

list(APPEND EXTRA_COMPONENT_DIRS
    "${CMAKE_CURRENT_LIST_DIR}/../git_common_components/esp-tflite-micro"
    "${CMAKE_CURRENT_LIST_DIR}/../git_common_components/esp-nn"
    "${CMAKE_CURRENT_LIST_DIR}/../git_common_components/esp_littlefs")

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(tomato_village_mobilenetv2_alpha1_esp32s3)
