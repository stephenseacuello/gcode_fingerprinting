/*
 * Arduino Nano 33 BLE Sense - CNC Sensor Logger
 * ==============================================
 * 
 * Reads sensor data and outputs JSON lines over Serial.
 * 
 * Sensors:
 * - LSM9DS1: 9-axis IMU (accelerometer + gyroscope + magnetometer)
 * - LPS22HB: Barometric pressure + temperature
 * - APDS9960: Proximity + gesture detection
 * - PDM Microphone: Audio RMS level
 * 
 * CSV Output Format:
 * id,timestamp,rms,ax,ay,az,gx,gy,gz,pressure,temp,proximity,gesture
 * 
 * Upload this to your Arduino Nano 33 BLE Sense boards
 * Set Serial baud rate to 115200
 * 
 * Author: Stephen Eacuello
 * Version: 2.0
 */

#include <Arduino_LSM9DS1.h>
#include <Arduino_LPS22HB.h>
#include <Arduino_APDS9960.h>
#include <PDM.h>

// Configuration
const int DEVICE_ID = 1;  // Change this for each Arduino board (1, 2, 3, etc.)
const int SAMPLE_RATE = 50;  // Hz - adjust based on your needs
const int SAMPLE_INTERVAL = 1000 / SAMPLE_RATE;  // milliseconds

// PDM audio buffer
const int AUDIO_BUFFER_SIZE = 256;
short audioBuffer[AUDIO_BUFFER_SIZE];
volatile int samplesRead = 0;

// Sensor data structure
struct SensorData {
  unsigned long timestamp;
  int rms;
  int ax, ay, az;      // Accelerometer in milli-g
  int gx, gy, gz;      // Gyroscope in milli-dps
  int pressure;        // Pressure in deci-hPa
  int temp;            // Temperature in centi-°C
  int proximity;       // 0-255
  int gesture;         // 0=none, 1=up, 2=down, 3=left, 4=right
};

// =================== SETUP ===================
void setup() {
  // Initialize serial
  Serial.begin(115200);
  while (!Serial && millis() < 3000);  // Wait up to 3 seconds for serial
  
  // Initialize IMU (Accelerometer + Gyroscope)
  if (!IMU.begin()) {
    Serial.println("ERROR: Failed to initialize IMU!");
    while (1);
  }
  
  // Initialize Pressure/Temperature sensor
  if (!BARO.begin()) {
    Serial.println("ERROR: Failed to initialize pressure sensor!");
    while (1);
  }
  
  // Initialize Proximity/Gesture sensor
  if (!APDS.begin()) {
    Serial.println("ERROR: Failed to initialize proximity sensor!");
    while (1);
  }
  
  // Initialize PDM microphone
  PDM.onReceive(onPDMdata);
  if (!PDM.begin(1, 16000)) {
    Serial.println("ERROR: Failed to start PDM!");
    while (1);
  }
  
  // Print header
  Serial.println("id,ts,rms,ax,ay,az,gx,gy,gz,pressure,temp,proximity,gesture");
  
  delay(100);
}

// =================== LOOP ===================
void loop() {
  static unsigned long lastSampleTime = 0;
  unsigned long currentTime = millis();
  
  // Sample at fixed rate
  if (currentTime - lastSampleTime >= SAMPLE_INTERVAL) {
    lastSampleTime = currentTime;
    
    SensorData data;
    data.timestamp = currentTime;
    
    // Read IMU
    readIMU(data);
    
    // Read Pressure/Temperature
    readPressure(data);
    
    // Read Proximity/Gesture
    readProximity(data);
    
    // Calculate audio RMS
    data.rms = calculateRMS();
    
    // Output CSV
    outputCSV(data);
  }
}

// =================== SENSOR READING FUNCTIONS ===================

void readIMU(SensorData &data) {
  float ax_g, ay_g, az_g;
  float gx_dps, gy_dps, gz_dps;
  
  if (IMU.accelerationAvailable()) {
    IMU.readAcceleration(ax_g, ay_g, az_g);
    
    // Convert to milli-g
    data.ax = (int)(ax_g * 1000.0);
    data.ay = (int)(ay_g * 1000.0);
    data.az = (int)(az_g * 1000.0);
  } else {
    data.ax = data.ay = data.az = 0;
  }
  
  if (IMU.gyroscopeAvailable()) {
    IMU.readGyroscope(gx_dps, gy_dps, gz_dps);
    
    // Convert to milli-degrees per second
    data.gx = (int)(gx_dps * 1000.0);
    data.gy = (int)(gy_dps * 1000.0);
    data.gz = (int)(gz_dps * 1000.0);
  } else {
    data.gx = data.gy = data.gz = 0;
  }
}

void readPressure(SensorData &data) {
  float pressure_kPa = BARO.readPressure();
  float temp_c = BARO.readTemperature();
  
  // Convert to deci-hPa (1 kPa = 10 hPa, so 1 kPa = 100 deci-hPa)
  data.pressure = (int)(pressure_kPa * 100.0);
  
  // Convert to centi-°C
  data.temp = (int)(temp_c * 100.0);
}

void readProximity(SensorData &data) {
  // Read proximity (0-255)
  if (APDS.proximityAvailable()) {
    data.proximity = APDS.readProximity();
  } else {
    data.proximity = 0;
  }
  
  // Read gesture
  data.gesture = 0;  // Default: no gesture
  
  if (APDS.gestureAvailable()) {
    int gesture = APDS.readGesture();
    
    switch (gesture) {
      case GESTURE_UP:
        data.gesture = 1;
        break;
      case GESTURE_DOWN:
        data.gesture = 2;
        break;
      case GESTURE_LEFT:
        data.gesture = 3;
        break;
      case GESTURE_RIGHT:
        data.gesture = 4;
        break;
      default:
        data.gesture = 0;
        break;
    }
  }
}

int calculateRMS() {
  if (samplesRead == 0) {
    return 0;
  }
  
  // Calculate RMS of audio samples
  long sum = 0;
  for (int i = 0; i < samplesRead; i++) {
    long sample = audioBuffer[i];
    sum += sample * sample;
  }
  
  samplesRead = 0;  // Reset for next calculation
  
  // Return RMS value (0-1023 range for Arduino compatibility)
  float rms = sqrt(sum / (float)AUDIO_BUFFER_SIZE);
  return constrain((int)(rms / 32), 0, 1023);
}

// =================== PDM CALLBACK ===================
void onPDMdata() {
  int bytesAvailable = PDM.available();
  PDM.read(audioBuffer, bytesAvailable);
  samplesRead = bytesAvailable / 2;  // 16-bit samples
}

// =================== OUTPUT ===================
void outputCSV(const SensorData &data) {
  // Format: id,timestamp,rms,ax,ay,az,gx,gy,gz,pressure,temp,proximity,gesture
  Serial.print(DEVICE_ID);
  Serial.print(",");
  Serial.print(data.timestamp);
  Serial.print(",");
  Serial.print(data.rms);
  Serial.print(",");
  Serial.print(data.ax);
  Serial.print(",");
  Serial.print(data.ay);
  Serial.print(",");
  Serial.print(data.az);
  Serial.print(",");
  Serial.print(data.gx);
  Serial.print(",");
  Serial.print(data.gy);
  Serial.print(",");
  Serial.print(data.gz);
  Serial.print(",");
  Serial.print(data.pressure);
  Serial.print(",");
  Serial.print(data.temp);
  Serial.print(",");
  Serial.print(data.proximity);
  Serial.print(",");
  Serial.println(data.gesture);
}

/*
 * NOTES:
 * 
 * 1. Change DEVICE_ID for each Arduino board you use (1, 2, 3, etc.)
 * 2. Adjust SAMPLE_RATE based on your application (10-100 Hz typical)
 * 3. Higher sample rates provide better resolution but generate more data
 * 4. Make sure your serial monitor is set to 115200 baud
 * 
 * SENSOR RANGES:
 * - Accelerometer: ±4g (default)
 * - Gyroscope: ±2000 dps (default)
 * - Pressure: 260-1260 hPa
 * - Temperature: -40 to +85°C
 * - Proximity: 0-255 (closer = higher value)
 * 
 * TROUBLESHOOTING:
 * - If sensors fail to initialize, check I2C connections
 * - If no data appears, check serial baud rate (115200)
 * - If gestures don't work, ensure nothing blocks proximity sensor
 * - For audio issues, check PDM microphone is enabled
 */
