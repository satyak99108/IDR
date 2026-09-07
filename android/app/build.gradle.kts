plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.idr.navigation"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.idr.navigation"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0.0-mvp"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
    aaptOptions {
        noCompress("tflite")
    }
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)

    // OpenStreetMap rendering (OSMDroid)
    implementation(libs.osmdroid.android)

    // TensorFlow Lite on-device inference
    implementation(libs.tensorflow.lite)
    implementation(libs.tensorflow.lite.support)

    // Coroutines for background replay and sensor streams
    implementation(libs.kotlinx.coroutines.android)

    // Unit tests
    testImplementation(libs.junit)
}
