#ifndef I2C_TM4C_H
#define I2C_TM4C_H

typedef unsigned char uint8;
typedef unsigned int uint32;
#define uint8_t uint8

// basic i2c setup -- needs to be called once at the beginning of the program
void i2c_common_setup(void);

// write value val to register address reg in I2C device with address addr
// the input argument flag is currently not used
int i2c_write_reg(uint32 addr, uint8_t reg, uint8_t val, int *flag);

// write n bytes starting from register address reg in I2C device with address addr
int i2c_write_bytes(uint32 addr, uint8_t reg, int n, uint8_t const *val, int *flag);

// read n bytes starting from register address reg in I2C device with address addr
int i2c_read_bytes(uint32 addr, uint8_t reg, int n, uint8_t *ret, int *flag);

#endif
