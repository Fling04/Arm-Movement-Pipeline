#include <Wire.h>
#include <SparkFun_BMI270_Arduino_Library.h>
#include <ADC.h>

#define IMU_HZ      400
#define FLEX_HZ     400
#define SERIAL_BAUD 115200
#define FLEX_EVERY  (IMU_HZ / FLEX_HZ)
#define FLEX_COUNT  5

const uint8_t FLEX_PINS[FLEX_COUNT] = {40, 41, 39, 25, 26};
const int FLEX_FLAT[FLEX_COUNT] = {2400, 2400, 2400, 2400, 2400};
const int FLEX_BENT[FLEX_COUNT] = {3500, 3500, 3500, 3500, 3500};

BMI270 imu;
ADC    adc;

uint32_t imu_count    = 0;
uint32_t flex_count   = 0;
uint32_t timing_slips = 0;
uint32_t last_report  = 0;

void setup_adc() {
  adc.adc0->setAveraging(4);
  adc.adc0->setResolution(12);
  adc.adc0->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
  adc.adc0->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
  adc.adc1->setAveraging(4);
  adc.adc1->setResolution(12);
  adc.adc1->setConversionSpeed(ADC_CONVERSION_SPEED::HIGH_SPEED);
  adc.adc1->setSamplingSpeed(ADC_SAMPLING_SPEED::HIGH_SPEED);
}

void read_flex(float *norm) {
  int raw[FLEX_COUNT];
  adc.adc0->startSingleRead(41);
  adc.adc1->startSingleRead(40);
  while (!adc.adc0->isComplete() || !adc.adc1->isComplete());
  raw[1] = adc.adc0->readSingle();
  raw[0] = adc.adc1->readSingle();
  adc.adc0->startSingleRead(25);
  adc.adc1->startSingleRead(39);
  while (!adc.adc0->isComplete() || !adc.adc1->isComplete());
  raw[3] = adc.adc0->readSingle();
  raw[2] = adc.adc1->readSingle();
  adc.adc1->startSingleRead(26);
  while (!adc.adc1->isComplete());
  raw[4] = adc.adc1->readSingle();
  for (int i = 0; i < FLEX_COUNT; i++) {
    norm[i] = constrain(
      (float)(raw[i] - FLEX_FLAT[i]) / (float)(FLEX_BENT[i] - FLEX_FLAT[i]),
      0.0f, 1.0f
    );
  }
}

void setup() {
  Serial.begin(SERIAL_BAUD);
  while (!Serial && millis() < 3000);
  delay(500);
  Wire.begin();
  Wire.setClock(400000);
  if (imu.beginI2C() != BMI2_OK) {
    Serial.println("# BMI270 not found!");
    while (1);
  }
  imu.setAccelODR(BMI2_ACC_ODR_400HZ);
  imu.setGyroODR(BMI2_GYR_ODR_400HZ);
  imu.setAccelPowerMode(BMI2_PERF_OPT_MODE);
  imu.setGyroPowerMode(BMI2_PERF_OPT_MODE, BMI2_PERF_OPT_MODE);
  setup_adc();
  Serial.println("# Setup done");
}

void loop() {
  static uint32_t next_us = micros();
  static uint8_t  tick    = 0;
  static float    flex_norm[FLEX_COUNT] = {0};

  uint32_t now = micros();
  if (now < next_us) return;
  if (now > next_us + (1000000 / IMU_HZ)) timing_slips++;
  next_us += (1000000 / IMU_HZ);

  imu.getSensorData();

  if (tick % FLEX_EVERY == 0) {
    read_flex(flex_norm);
    flex_count++;
  }

  // print everything as a comma separated stream
  Serial.printf("%lu,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
    now,
    imu.data.accelX, imu.data.accelY, imu.data.accelZ,
    imu.data.gyroX,  imu.data.gyroY,  imu.data.gyroZ,
    flex_norm[0], flex_norm[1], flex_norm[2], flex_norm[3], flex_norm[4]
  );

  imu_count++;
  tick++;
  if (tick >= 255) tick = 0;

  if ((now - last_report > 5000000) && (flex_norm[0]>0) && (flex_norm[1]>0) && (flex_norm[2]>0)&& (flex_norm[3]>0)&& (flex_norm[4]>0)) {
    Serial.printf("# IMU: %.0fHz | Flex: %.0fHz | Slips: %lu\n"
      imu_count / 5.0f, flex_count / 5.0f, timing_slips);
    imu_count = flex_count = timing_slips = 0;
    last_report = now;
  }
}