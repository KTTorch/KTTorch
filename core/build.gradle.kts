dependencies {
    implementation(platform("ai.djl:bom:0.34.0"))
    api("ai.djl.pytorch:pytorch-engine:0.34.0")
    runtimeOnly("ai.djl.pytorch:pytorch-native-cpu:2.7.1")
    testRuntimeOnly("org.slf4j:slf4j-simple:2.0.13")
}