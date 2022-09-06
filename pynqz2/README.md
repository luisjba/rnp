## Writting ans SD Card Image for XILINX PYNQ Z2

This installation is only for Mac OS

1 - Download the image , latest version , in this tutorial, we downloaded the 2.7.0 version. Search fot he version of
images [here](http://www.pynq.io/board.html)

- Opent the Terminal and use the `diskutil` to find four your sd card disk identification to then unmount it.
```bash
diskutil list

/dev/disk0 (internal):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:      GUID_partition_scheme                         1.0 TB     disk0
   1:             Apple_APFS_ISC ⁨⁩                        524.3 MB   disk0s1
   2:                 Apple_APFS ⁨Container disk3⁩         994.7 GB   disk0s2
   3:        Apple_APFS_Recovery ⁨⁩                        5.4 GB     disk0s3

/dev/disk3 (synthesized):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:      APFS Container Scheme -                      +994.7 GB   disk3
                                 Physical Store disk0s2
   1:                APFS Volume ⁨Macintosh HD⁩            15.2 GB    disk3s1
   2:              APFS Snapshot ⁨com.apple.os.update-...⁩ 15.2 GB    disk3s1s1
   3:                APFS Volume ⁨Preboot⁩                 464.9 MB   disk3s2
   4:                APFS Volume ⁨Recovery⁩                799.4 MB   disk3s3
   5:                APFS Volume ⁨Data⁩                    167.0 GB   disk3s5
   6:                APFS Volume ⁨VM⁩                      20.5 KB    disk3s6

/dev/disk4 (internal, physical):
   #:                       TYPE NAME                    SIZE       IDENTIFIER
   0:     FDisk_partition_scheme                        *62.3 GB    disk4
   1:             Windows_FAT_32 ⁨boot⁩                    268.4 MB   disk4s1
   2:                      Linux ⁨⁩                        62.0 GB    disk4s2
```

For our example, as we inserted a 64 GB sd card, is the last device identifier `disk4`

- Unmount the SD card by using th disk identifier `disk4`(this only apply for the computer that I used whn building this tutorial).
```bash
diskutil unmountDisk /dev/disk4

Unmount of all volumes on disk4 was successful
```

- Copy the data into your SD Card. Locate the image directory, mine 
is located in `/Users/luisjba/Downloads/pynq_z2_v2.7.0.img`. Do not forget to provide full path, avoid relative path.
```bash
sudo dd bs=1m if=/Users/luisjba/Downloads/pynq_z2_v2.7.0.img of=/dev/rdisk4
```

After executing the previous command, you have to wait until finish, I take some time depend on the 
machine and the speed of writing into the SD card.

- Once the write process is finished , unmount the SD Card
```bash
diskutil unmountDisk /dev/disk4
```
- take the SD card and insert into the PYNQ-Z2 board.


# Install HSL4 ML

```bash
python -m pip install "hls4ml"
python -m pip install "hls4ml[profiling]"
```

# Installing Vivado in Docker

We have to download the latest Vivado ML Edition. Four our installation we have downloaded the version 2021.2 from here 
[Download Vivado ML 2021.2 ](https://www.xilinx.com/member/forms/download/xef.html?filename=Xilinx_Unified_2021.2_1021_0703.tar.gz), you will need an account. Te file name should be `Xilinx_Unified_2021.2_1021_0703.tar.gz`.

