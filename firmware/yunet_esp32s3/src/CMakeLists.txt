idf_component_register(
    SRCS "main.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_http_server esp_wifi esp_netif esp_event nvs_flash esp-tflite-micro
    PRIV_REQUIRES esp_littlefs)
