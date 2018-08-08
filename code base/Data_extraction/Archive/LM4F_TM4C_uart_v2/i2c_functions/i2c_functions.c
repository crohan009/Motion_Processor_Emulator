#include "i2c_functions.h"
#include "i2c_TM4C.h"


int i2c_write(unsigned char slave_addr, unsigned char reg_addr,
		unsigned char length, unsigned char const *data)
{
	if (length < 2)
	{
		int i;
		for (i=0; i<length; i++)
		{
			i2c_write_reg(slave_addr, reg_addr + i, data[i], 0);
		}
	}
	else
		i2c_write_bytes(slave_addr,reg_addr,length,data,0);
	return 0;
}

int i2c_read(unsigned char slave_addr, unsigned char reg_addr,
		unsigned char length, unsigned char *data)
{
	i2c_read_bytes(slave_addr, reg_addr, length, data, 0);
	return 0;
}

int i2c_write_lsm(unsigned char slave_addr, unsigned char reg_addr,
        unsigned char length, unsigned char const *data)
{
    if (length < 2)
    {
        int i;
        for (i=0; i<length; i++)
        {
            i2c_write_reg(slave_addr, reg_addr + i, data[i], 0);
        }
    }
    else
        i2c_write_bytes(slave_addr,reg_addr,length,data,0);
    return 1;
}

int i2c_read_lsm(unsigned char slave_addr, unsigned char reg_addr,
        unsigned char length, unsigned char *data)
{
    i2c_read_bytes(slave_addr, reg_addr, length, data, 0);
    return 1;
}
