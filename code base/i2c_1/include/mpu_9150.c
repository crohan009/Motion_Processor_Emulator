#include "mpu_9150.h"
#include "math.h"
#include "i2c_tm4c.h"

int *f;
int i;
float aRes, gRes, mRes; // scale resolutions per LSB for the sensors

float GyroMeasError = PI * (40.0f / 180.0f);   // gyroscope measurement error in rads/s (start at 40 deg/s)
float GyroMeasDrift = PI * (0.0f  / 180.0f);   // gyroscope measurement drift in rad/s/s (start at 0.0 deg/s/s)

//float beta = sqrt(3.0f / 4.0f) * GyroMeasError;   // compute beta
//float zeta = sqrt(3.0f / 4.0f) * GyroMeasDrift;   // compute zeta, the other free parameter in the Madgwick scheme usually set to a small or zero value


// Set initial input parameters
enum Ascale {
  AFS_2G = 0,
  AFS_4G,
  AFS_8G,
  AFS_16G
};

enum Gscale {
  GFS_250DPS = 0,
  GFS_500DPS,
  GFS_1000DPS,
  GFS_2000DPS
};

// Specify sensor full scale
uint8_t Gscale = GFS_250DPS;
uint8_t Ascale = AFS_2G;

void delay (unsigned  long time  )
{
    SysCtlDelay(time * (SysCtlClockGet() / 3 / 1000));
}

void MPU9150SelfTest(float * destination) // Should return percent deviation from factory trim values, +/- 14 or less deviation is a pass
{
   uint8_t rawData[4];
   uint8_t selfTest[6];
   float factoryTrim[6];

   // Configure the accelerometer for self-test
   i2c_write_reg(MPU9150_ADDRESS, ACCEL_CONFIG, 0xF0,f); // Enable self test on all three axes and set accelerometer range to +/- 8 g
   i2c_write_reg(MPU9150_ADDRESS, GYRO_CONFIG, 0xE0,f); // Enable self test on all three axes and set gyro range to +/- 250 degrees/s
   delay(250);  // Delay a while to let the device execute the self-test
   rawData[0] = I2CReceive(MPU9150_ADDRESS, SELF_TEST_X); // X-axis self-test results
   rawData[1] = I2CReceive(MPU9150_ADDRESS, SELF_TEST_Y); // Y-axis self-test results
   rawData[2] = I2CReceive(MPU9150_ADDRESS, SELF_TEST_Z); // Z-axis self-test results
   rawData[3] = I2CReceive(MPU9150_ADDRESS, SELF_TEST_A); // Mixed-axis self-test results
   // Extract the acceleration test results first
   selfTest[0] = (rawData[0] >> 3) | (rawData[3] & 0x30) >> 4 ; // XA_TEST result is a five-bit unsigned integer
   selfTest[1] = (rawData[1] >> 3) | (rawData[3] & 0x0C) >> 4 ; // YA_TEST result is a five-bit unsigned integer
   selfTest[2] = (rawData[2] >> 3) | (rawData[3] & 0x03) >> 4 ; // ZA_TEST result is a five-bit unsigned integer
   // Extract the gyration test results first
   selfTest[3] = rawData[0]  & 0x1F ; // XG_TEST result is a five-bit unsigned integer
   selfTest[4] = rawData[1]  & 0x1F ; // YG_TEST result is a five-bit unsigned integer
   selfTest[5] = rawData[2]  & 0x1F ; // ZG_TEST result is a five-bit unsigned integer
   // Process results to allow final comparison with factory set values
   factoryTrim[0] = (4096.0*0.34)*(pow( (0.92/0.34) , (((float)selfTest[0] - 1.0)/30.0))); // FT[Xa] factory trim calculation
   factoryTrim[1] = (4096.0*0.34)*(pow( (0.92/0.34) , (((float)selfTest[1] - 1.0)/30.0))); // FT[Ya] factory trim calculation
   factoryTrim[2] = (4096.0*0.34)*(pow( (0.92/0.34) , (((float)selfTest[2] - 1.0)/30.0))); // FT[Za] factory trim calculation
   factoryTrim[3] =  ( 25.0*131.0)*(pow( 1.046 , ((float)selfTest[3] - 1.0) ));             // FT[Xg] factory trim calculation
   factoryTrim[4] =  (-25.0*131.0)*(pow( 1.046 , ((float)selfTest[4] - 1.0) ));             // FT[Yg] factory trim calculation
   factoryTrim[5] =  ( 25.0*131.0)*(pow( 1.046 , ((float)selfTest[5] - 1.0) ));             // FT[Zg] factory trim calculation


 // Report results as a ratio of (STR - FT)/FT; the change from Factory Trim of the Self-Test Response
 // To get to percent, must multiply by 100 and subtract result from 100

   for (i = 0; i < 6; i++) {
     destination[i] = 100.0 + 100.0*((float)selfTest[i] - factoryTrim[i])/factoryTrim[i]; // Report percent differences
   }

}


void initMPU9150()
{
 // wake up device
  i2c_write_reg(MPU9150_ADDRESS, PWR_MGMT_1, 0x00,f); // Clear sleep mode bit (6), enable all sensors
  delay(100); // Delay 100 ms for PLL to get established on x-axis gyro; should check for PLL ready interrupt

 // get stable time source
  i2c_write_reg(MPU9150_ADDRESS, PWR_MGMT_1, 0x01,f);  // Set clock source to be PLL with x-axis gyroscope reference, bits 2:0 = 001
  delay(200);

 // Configure Gyro and Accelerometer
 // Disable FSYNC and set accelerometer and gyro bandwidth to 44 and 42 Hz, respectively;
 // DLPF_CFG = bits 2:0 = 010; this sets the sample rate at 1 kHz for both
 // Minimum delay time is 4.9 ms which sets the fastest rate at ~200 Hz
//  i2c_write_reg(MPU9150_ADDRESS, CONFIG, 0x03,f);
  i2c_write_reg(MPU9150_ADDRESS, CONFIG, 0x00,f);

 // Set sample rate = gyroscope output rate/(1 + SMPLRT_DIV)
//  i2c_write_reg(MPU9150_ADDRESS, SMPLRT_DIV, 0x04,f);  // Use a 200 Hz rate; the same rate set in CONFIG above
  i2c_write_reg(MPU9150_ADDRESS, SMPLRT_DIV, 0x07,f);

 // Set gyroscope full scale range
 // Range selects FS_SEL and AFS_SEL are 0 - 3, so 2-bit values are left-shifted into positions 4:3
  uint8_t c =  I2CReceive(MPU9150_ADDRESS, GYRO_CONFIG);
  i2c_write_reg(MPU9150_ADDRESS, GYRO_CONFIG, c & ~0xE0,f); // Clear self-test bits [7:5]
  i2c_write_reg(MPU9150_ADDRESS, GYRO_CONFIG, c & ~0x18,f); // Clear AFS bits [4:3]
  i2c_write_reg(MPU9150_ADDRESS, GYRO_CONFIG, c | Gscale << 3,f); // Set full scale range for the gyro

 // Set accelerometer configuration
  c =  I2CReceive(MPU9150_ADDRESS, ACCEL_CONFIG);
  i2c_write_reg(MPU9150_ADDRESS, ACCEL_CONFIG, c & ~0xE0,f); // Clear self-test bits [7:5]
  i2c_write_reg(MPU9150_ADDRESS, ACCEL_CONFIG, c & ~0x18,f); // Clear AFS bits [4:3]
  i2c_write_reg(MPU9150_ADDRESS, ACCEL_CONFIG, c | Ascale << 3,f); // Set full scale range for the accelerometer

 
 // Configure Magnetometer for FIFO
 // Initialize AK8975A for write
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV1_ADDR, 0x0C,f);  // Write address of AK8975A
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV1_REG, 0x0A,f);   // Register from within the AK8975 to which to write
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV1_DO, 0x01,f);    // Register that holds output data written into Slave 1 when in write mode
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV1_CTRL, 0x81,f);  // Enable Slave 1

 // Set up auxilliary communication with AK8975A for FIFO read
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV0_ADDR, 0x8C,f); // Enable and read address (0x0C) of the AK8975A
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV0_REG, 0x03,f);  // Register within AK8975A from which to start data read
   i2c_write_reg(MPU9150_ADDRESS, I2C_SLV0_CTRL, 0x86,f); // Read six bytes and swap bytes

 // Configure FIFO
   i2c_write_reg(MPU9150_ADDRESS, INT_ENABLE, 0x00,f); // Disable all interrupts
   i2c_write_reg(MPU9150_ADDRESS, FIFO_EN, 0x00,f);    // Disable FIFO
   i2c_write_reg(MPU9150_ADDRESS, USER_CTRL, 0x02,f);  // Reset I2C master and FIFO and DMP
   i2c_write_reg(MPU9150_ADDRESS, USER_CTRL, 0x00,f);  // Disable FIFO
   delay(100);
   i2c_write_reg(MPU9150_ADDRESS, FIFO_EN, 0xF9,f); // Enable all sensors for FIFO
   i2c_write_reg(MPU9150_ADDRESS, I2C_MST_DELAY_CTRL, 0x80,f); // Enable delay of external sensor data until all data registers have been read


  // Configure Interrupts and Bypass Enable
  // Set interrupt pin active high, push-pull, and clear on read of INT_STATUS, enable I2C_BYPASS_EN so additional chips
  // can join the I2C bus and all can be controlled by the Arduino as master
   i2c_write_reg(MPU9150_ADDRESS, INT_PIN_CFG, 0x22,f);
   i2c_write_reg(MPU9150_ADDRESS, INT_ENABLE, 0x01,f);  // Enable data ready (bit 0) interrupt
}


void calibrateMPU9150(float * dest1, float * dest2)
{
  uint8_t data[12]; // data array to hold accelerometer and gyro x, y, z, data
  uint16_t ii, packet_count, fifo_count;
  int32_t gyro_bias[3]  = {0, 0, 0}, accel_bias[3] = {0, 0, 0};

// reset device, reset all registers, clear gyro and accelerometer bias registers
  i2c_write_reg(MPU9150_ADDRESS, PWR_MGMT_1, 0x80,f); // Write a one to bit 7 reset bit; toggle reset device
  delay(100);

// get stable time source
// Set clock source to be PLL with x-axis gyroscope reference, bits 2:0 = 001
  i2c_write_reg(MPU9150_ADDRESS, PWR_MGMT_1, 0x01,f);
  i2c_write_reg(MPU9150_ADDRESS, PWR_MGMT_2, 0x00,f);
  delay(200);

// Configure device for bias calculation
  i2c_write_reg(MPU9150_ADDRESS, INT_ENABLE, 0x00,f);   // Disable all interrupts
  i2c_write_reg(MPU9150_ADDRESS, FIFO_EN, 0x00,f);      // Disable FIFO
  i2c_write_reg(MPU9150_ADDRESS, PWR_MGMT_1, 0x00,f);   // Turn on internal clock source
  i2c_write_reg(MPU9150_ADDRESS, I2C_MST_CTRL, 0x00,f); // Disable I2C master
  i2c_write_reg(MPU9150_ADDRESS, USER_CTRL, 0x00,f);    // Disable FIFO and I2C master modes
  i2c_write_reg(MPU9150_ADDRESS, USER_CTRL, 0x0C,f);    // Reset FIFO and DMP
  delay(150);

// Configure MPU6050 gyro and accelerometer for bias calculation
  i2c_write_reg(MPU9150_ADDRESS, CONFIG, 0x01,f);      // Set low-pass filter to 188 Hz
  i2c_write_reg(MPU9150_ADDRESS, SMPLRT_DIV, 0x00,f);  // Set sample rate to 1 kHz
  i2c_write_reg(MPU9150_ADDRESS, GYRO_CONFIG, 0x00,f);  // Set gyro full-scale to 250 degrees per second, maximum sensitivity
  i2c_write_reg(MPU9150_ADDRESS, ACCEL_CONFIG, 0x00,f); // Set accelerometer full-scale to 2 g, maximum sensitivity

  uint16_t  gyrosensitivity  = 131;   // = 131 LSB/degrees/sec
  uint16_t  accelsensitivity = 16384;  // = 16384 LSB/g

// Configure FIFO to capture accelerometer and gyro data for bias calculation
  i2c_write_reg(MPU9150_ADDRESS, USER_CTRL, 0x40,f);   // Enable FIFO
  i2c_write_reg(MPU9150_ADDRESS, FIFO_EN, 0x78,f);     // Enable gyro and accelerometer sensors for FIFO  (max size 1024 bytes in MPU-6050)
  delay(80); // accumulate 80 samples in 80 milliseconds = 960 bytes

// At end of sample accumulation, turn off FIFO sensor read
  i2c_write_reg(MPU9150_ADDRESS, FIFO_EN, 0x00,f);        // Disable gyro and accelerometer sensors for FIFO
  data[0] = I2CReceive(MPU9150_ADDRESS, FIFO_COUNTH); // read FIFO sample count
  data[1] = I2CReceive(MPU9150_ADDRESS, FIFO_COUNTL);
  fifo_count = ((uint16_t)data[0] << 8) | data[1];
  packet_count = fifo_count/12;// How many sets of full gyro and accelerometer data for averaging

     for (ii = 0; ii < packet_count; ii++) {
        int16_t accel_temp[3] = {0, 0, 0}, gyro_temp[3] = {0, 0, 0};
        data[0]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W); // read data for averaging
        data[1]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[2]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[3]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[4]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[5]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[6]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[7]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[8]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[9]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[10]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        data[11]=I2CReceive(MPU9150_ADDRESS, FIFO_R_W);
        accel_temp[0] = (int16_t) (((int16_t)data[0] << 8) | data[1]  ) ;  // Form signed 16-bit integer for each sample in FIFO
        accel_temp[1] = (int16_t) (((int16_t)data[2] << 8) | data[3]  ) ;
        accel_temp[2] = (int16_t) (((int16_t)data[4] << 8) | data[5]  ) ;
        gyro_temp[0]  = (int16_t) (((int16_t)data[6] << 8) | data[7]  ) ;
        gyro_temp[1]  = (int16_t) (((int16_t)data[8] << 8) | data[9]  ) ;
        gyro_temp[2]  = (int16_t) (((int16_t)data[10] << 8) | data[11]) ;

        accel_bias[0] += (int32_t) accel_temp[0]; // Sum individual signed 16-bit biases to get accumulated signed 32-bit biases
        accel_bias[1] += (int32_t) accel_temp[1];
        accel_bias[2] += (int32_t) accel_temp[2];
        gyro_bias[0]  += (int32_t) gyro_temp[0];
        gyro_bias[1]  += (int32_t) gyro_temp[1];
        gyro_bias[2]  += (int32_t) gyro_temp[2];

     }

    accel_bias[0] /= (int32_t) packet_count; // Normalize sums to get average count biases
    accel_bias[1] /= (int32_t) packet_count;
    accel_bias[2] /= (int32_t) packet_count;
    gyro_bias[0]  /= (int32_t) packet_count;
    gyro_bias[1]  /= (int32_t) packet_count;
    gyro_bias[2]  /= (int32_t) packet_count;

  if(accel_bias[2] > 0L) {accel_bias[2] -= (int32_t) accelsensitivity;}  // Remove gravity from the z-axis accelerometer bias calculation
  else {accel_bias[2] += (int32_t) accelsensitivity;}

// Construct the gyro biases for push to the hardware gyro bias registers, which are reset to zero upon device startup
  data[0] = (-gyro_bias[0]/4  >> 8) & 0xFF; // Divide by 4 to get 32.9 LSB per deg/s to conform to expected bias input format
  data[1] = (-gyro_bias[0]/4)       & 0xFF; // Biases are additive, so change sign on calculated average gyro biases
  data[2] = (-gyro_bias[1]/4  >> 8) & 0xFF;
  data[3] = (-gyro_bias[1]/4)       & 0xFF;
  data[4] = (-gyro_bias[2]/4  >> 8) & 0xFF;
  data[5] = (-gyro_bias[2]/4)       & 0xFF;

// Push gyro biases to hardware registers
  i2c_write_reg(MPU9150_ADDRESS, XG_OFFS_USRH, data[0],f);
  i2c_write_reg(MPU9150_ADDRESS, XG_OFFS_USRL, data[1],f);
  i2c_write_reg(MPU9150_ADDRESS, YG_OFFS_USRH, data[2],f);
  i2c_write_reg(MPU9150_ADDRESS, YG_OFFS_USRL, data[3],f);
  i2c_write_reg(MPU9150_ADDRESS, ZG_OFFS_USRH, data[4],f);
  i2c_write_reg(MPU9150_ADDRESS, ZG_OFFS_USRL, data[5],f);

// Output scaled gyro biases for display in the main program
  dest1[0] = (float) gyro_bias[0]/(float) gyrosensitivity;
  dest1[1] = (float) gyro_bias[1]/(float) gyrosensitivity;
  dest1[2] = (float) gyro_bias[2]/(float) gyrosensitivity;

// Construct the accelerometer biases for push to the hardware accelerometer bias registers. These registers contain
// factory trim values which must be added to the calculated accelerometer biases; on boot up these registers will hold
// non-zero values. In addition, bit 0 of the lower byte must be preserved since it is used for temperature
// compensation calculations. Accelerometer bias registers expect bias input as 2048 LSB per g, so that
// the accelerometer biases calculated above must be divided by 8.

  int32_t accel_bias_reg[3] = {0, 0, 0}; // A place to hold the factory accelerometer trim biases
  data[0]=I2CReceive(MPU9150_ADDRESS, XA_OFFSET_H); // Read factory accelerometer trim values
  data[1]=I2CReceive(MPU9150_ADDRESS, XA_OFFSET_L_TC);
  accel_bias_reg[0] = (int16_t) ((int16_t)data[0] << 8) | data[1];
  data[0]=I2CReceive(MPU9150_ADDRESS, YA_OFFSET_H); // Read factory accelerometer trim values
  data[1]=I2CReceive(MPU9150_ADDRESS, YA_OFFSET_L_TC);
  accel_bias_reg[1] = (int16_t) ((int16_t)data[0] << 8) | data[1];
  data[0]=I2CReceive(MPU9150_ADDRESS, ZA_OFFSET_H); // Read factory accelerometer trim values
  data[1]=I2CReceive(MPU9150_ADDRESS, ZA_OFFSET_L_TC);
  accel_bias_reg[2] = (int16_t) ((int16_t)data[0] << 8) | data[1];

  uint32_t mask = 1uL; // Define mask for temperature compensation bit 0 of lower byte of accelerometer bias registers
  uint8_t mask_bit[3] = {0, 0, 0}; // Define array to hold mask bit for each accelerometer bias axis

  for(ii = 0; ii < 3; ii++) {
    if(accel_bias_reg[ii] & mask) mask_bit[ii] = 0x01; // If temperature compensation bit is set, record that fact in mask_bit
  }

  // Construct total accelerometer bias, including calculated average accelerometer bias from above
  accel_bias_reg[0] -= (accel_bias[0]/8); // Subtract calculated averaged accelerometer bias scaled to 2048 LSB/g (16 g full scale)
  accel_bias_reg[1] -= (accel_bias[1]/8);
  accel_bias_reg[2] -= (accel_bias[2]/8);

  data[0] = (accel_bias_reg[0] >> 8) & 0xFF;
  data[1] = (accel_bias_reg[0])      & 0xFF;
  data[1] = data[1] | mask_bit[0]; // preserve temperature compensation bit when writing back to accelerometer bias registers
  data[2] = (accel_bias_reg[1] >> 8) & 0xFF;
  data[3] = (accel_bias_reg[1])      & 0xFF;
  data[3] = data[3] | mask_bit[1]; // preserve temperature compensation bit when writing back to accelerometer bias registers
  data[4] = (accel_bias_reg[2] >> 8) & 0xFF;
  data[5] = (accel_bias_reg[2])      & 0xFF;
  data[5] = data[5] | mask_bit[2]; // preserve temperature compensation bit when writing back to accelerometer bias registers

  // Push accelerometer biases to hardware registers
  i2c_write_reg(MPU9150_ADDRESS, XA_OFFSET_H, data[0],f);
  i2c_write_reg(MPU9150_ADDRESS, XA_OFFSET_L_TC, data[1],f);
  i2c_write_reg(MPU9150_ADDRESS, YA_OFFSET_H, data[2],f);
  i2c_write_reg(MPU9150_ADDRESS, YA_OFFSET_L_TC, data[3],f);
  i2c_write_reg(MPU9150_ADDRESS, ZA_OFFSET_H, data[4],f);
  i2c_write_reg(MPU9150_ADDRESS, ZA_OFFSET_L_TC, data[5],f);

// Output scaled accelerometer biases for display in the main program
   dest2[0] = (float)accel_bias[0]/(float)accelsensitivity;
   dest2[1] = (float)accel_bias[1]/(float)accelsensitivity;
   dest2[2] = (float)accel_bias[2]/(float)accelsensitivity;
}

void readAccelData(int16_t * destination)
{
  uint8_t rawData[6];  // x/y/z accel register data stored here
  rawData[0]=I2CReceive(MPU9150_ADDRESS, ACCEL_XOUT_H);  // Read the six raw data registers into data array
  rawData[1]=I2CReceive(MPU9150_ADDRESS, ACCEL_XOUT_L);
  rawData[2]=I2CReceive(MPU9150_ADDRESS, ACCEL_YOUT_H);
  rawData[3]=I2CReceive(MPU9150_ADDRESS, ACCEL_YOUT_L);
  rawData[4]=I2CReceive(MPU9150_ADDRESS, ACCEL_ZOUT_H);
  rawData[5]=I2CReceive(MPU9150_ADDRESS, ACCEL_ZOUT_L);
  destination[0] = ((int16_t)rawData[0] << 8) | rawData[1] ;  // Turn the MSB and LSB into a signed 16-bit value
  destination[1] = ((int16_t)rawData[2] << 8) | rawData[3] ;
  destination[2] = ((int16_t)rawData[4] << 8) | rawData[5] ;
}


void readGyroData(int16_t * destination)
{
  uint8_t rawData[6];  // x/y/z gyro register data stored here
  rawData[0]=I2CReceive(MPU9150_ADDRESS, GYRO_XOUT_H);  // Read the six raw data registers sequentially into data array
  rawData[1]=I2CReceive(MPU9150_ADDRESS, GYRO_XOUT_L);
  rawData[2]=I2CReceive(MPU9150_ADDRESS, GYRO_YOUT_H);
  rawData[3]=I2CReceive(MPU9150_ADDRESS, GYRO_YOUT_L);
  rawData[4]=I2CReceive(MPU9150_ADDRESS, GYRO_ZOUT_H);
  rawData[5]=I2CReceive(MPU9150_ADDRESS, GYRO_ZOUT_L);
  destination[0] = ((int16_t)rawData[0] << 8) | rawData[1] ;  // Turn the MSB and LSB into a signed 16-bit value
  destination[1] = ((int16_t)rawData[2] << 8) | rawData[3] ;
  destination[2] = ((int16_t)rawData[4] << 8) | rawData[5] ;
}

void readMagData(int16_t * destination)
{
  uint8_t rawData[6];  // x/y/z gyro register data stored here
  i2c_write_reg(AK8975A_ADDRESS, AK8975A_CNTL, 0x01, f); // toggle enable data read from magnetometer, no continuous read mode!
  delay(10);
  // Only accept a new magnetometer data read if the data ready bit is set and
  // if there are no sensor overflow or data read errors
  if(I2CReceive(AK8975A_ADDRESS, AK8975A_ST1) & 0x01) { // wait for magnetometer data ready bit to be set
      rawData[0] = I2CReceive(AK8975A_ADDRESS, AK8975A_XOUT_L);  // Read the six raw data registers sequentially into data array
      rawData[1] = I2CReceive(AK8975A_ADDRESS, AK8975A_XOUT_H);
      rawData[2] = I2CReceive(AK8975A_ADDRESS, AK8975A_YOUT_L);
      rawData[3] = I2CReceive(AK8975A_ADDRESS, AK8975A_YOUT_H);
      rawData[4] = I2CReceive(AK8975A_ADDRESS, AK8975A_ZOUT_L);
      rawData[5] = I2CReceive(AK8975A_ADDRESS, AK8975A_ZOUT_H);
      destination[0] = ((int16_t)rawData[1] << 8) | rawData[0] ;  // Turn the MSB and LSB into a signed 16-bit value
      destination[1] = ((int16_t)rawData[3] << 8) | rawData[2] ;
      destination[2] = ((int16_t)rawData[5] << 8) | rawData[4] ;
  }
}

void initAK8975A(float * destination)
{
  uint8_t rawData[3];  // x/y/z gyro register data stored here
  i2c_write_reg(AK8975A_ADDRESS, AK8975A_CNTL, 0x00,f); // Power down
  delay(10);
  i2c_write_reg(AK8975A_ADDRESS, AK8975A_CNTL, 0x0F,f); // Enter Fuse ROM access mode
  delay(10);
  rawData[0]=I2CReceive(AK8975A_ADDRESS, AK8975A_ASAX);  // Read the x-, y-, and z-axis calibration values
  rawData[1]=I2CReceive(AK8975A_ADDRESS, AK8975A_ASAY);
  rawData[2]=I2CReceive(AK8975A_ADDRESS, AK8975A_ASAZ);
  destination[0] =  (float)(rawData[0] - 128)/256. + 1.; // Return x-axis sensitivity adjustment values
  destination[1] =  (float)(rawData[1] - 128)/256. + 1.;
  destination[2] =  (float)(rawData[2] - 128)/256. + 1.;
}

int16_t readTempData()
{
  uint8_t rawData[2];  // x/y/z gyro register data stored here
  rawData[0] = I2CReceive(MPU9150_ADDRESS,TEMP_OUT_H);  // Read the two raw data registers sequentially into data array
  rawData[1] = I2CReceive(MPU9150_ADDRESS,TEMP_OUT_L);
  return ((int16_t)rawData[0] << 8) | rawData[1] ;  // Turn the MSB and LSB into a 16-bit value
}

float getGres() {
  switch (Gscale)
  {
    // Possible gyro scales (and their register bit settings) are:
    // 250 DPS (00), 500 DPS (01), 1000 DPS (10), and 2000 DPS  (11).
        // Here's a bit of an algorith to calculate DPS/(ADC tick) based on that 2-bit value:
    case GFS_250DPS:
          gRes = 250.0/32768.0;
          break;
    case GFS_500DPS:
          gRes = 500.0/32768.0;
          break;
    case GFS_1000DPS:
          gRes = 1000.0/32768.0;
          break;
    case GFS_2000DPS:
          gRes = 2000.0/32768.0;
          break;
  }
  return gRes;
}

float getAres() {
  switch (Ascale)
  {
    // Possible accelerometer scales (and their register bit settings) are:
    // 2 Gs (00), 4 Gs (01), 8 Gs (10), and 16 Gs  (11).
        // Here's a bit of an algorith to calculate DPS/(ADC tick) based on that 2-bit value:
    case AFS_2G:
          aRes = 2.0/32768.0;
          break;
    case AFS_4G:
          aRes = 4.0/32768.0;
          break;
    case AFS_8G:
          aRes = 8.0/32768.0;
          break;
    case AFS_16G:
          aRes = 16.0/32768.0;
          break;
  }
  return aRes;
}


