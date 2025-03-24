# Android

Resources for my android devices.
All of which are currently samsung galaxy devices.


## Apps

### Sources

- FDroid
- Aurora Store
- GH:
  - Look for stuff in "Kotlin" (such as trending, etc.)
  - maybe topics like: `android`, `android-app`, `fdroid`, `apk`, `magisk`,
    etc.

New one:
https://www.openapk.net/

### APKs
(What actual apps I use/get)

#### Normal

- newpipe
- antennapod
- Feeder
- VLC
- Fennec
- Brave
- DAVx5
- Nextcloud
- Amaze
- Joplin
- lawnchair: https://github.com/LawnchairLauncher/lawnchair


- LINE
- Mega
- OneNote
- Goog: maps, gmail, calendar, Keep, drive
-
-


#### Root

- magisk
- Viper4Android
- BCR
- App Manager


## Firmware

### `samloader`

https://github.com/samloader/samloader

> Download firmware for Samsung devices (without any extra Windows drivers).

Nice!

Install:

```sh
pip install git+https://github.com/samloader/samloader.git
```

Basic usage:

```sh
# Check the latest firmware version.
samloader -m SM-N9600 -r GTO checkupdate
# N9600ZHU9FVH2/N9600OWO9FVF3/N9600ZCU9FVH2/N9600ZHU9FVH2
```

Okay, nvm, I guess you can only download latest fm with samloader (same as frija).
I guess this is not an software limitation, but that samsung doesn't expose older versions maybe.





----



# Stuck In Prenormal?



... Try out a different CSC?



### Sammobile



https://www.sammobile.com/samsung/galaxy-note9/firmware/#SM-N9600



By **DEFAULT** your CSC is `TPA`.

"Panama" I guess?



#### CSC: SM-N9600 "Unknown" 'GTO'



LATEST:

- 2022-09-20:
  
   - Version: 10
  
   - Security Patch: `2022-06-01`
  
   - PDA: `N9600ZHU9FVH2`
  
   - CSC: `N9600OWO9FVF3`
  
   - Build Date: `Wed, 03 Aug 2022 15:15:44 +0000`

https://www.sammobile.com/samsung/galaxy-note9/firmware/SM-N9600/GTO/



OLDEST: `2021-12-23` ? (10)



#### CSC: SM-N9600 "Unknown" 'MXO' 💯

WAY MORE FIRMWARE THAN `GTO`!

USE THIS!


-----------


### ZFOLD5

```
./odin4 \
-b ZFOLD5/BL_F946BXXS1AWH3_F946BXXS1AWH3_MQB69133266_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a ZFOLD5/magisk_patched-26300_gZQfi.tar \
-c ZFOLD5/CP_F946BXXU1AWG3_CP24524237_MQB67483245_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-s ZFOLD5/CSC_OXM_F946BOXM1AWG4_MQB67565539_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/016
```

### SM-T970 Tab S7+

- patched magisk method
- OEM unlocking done
- Used Bifrost to get latest for EUY (EU generic) region of SM-T970
- odin4: https://forum.xda-developers.com/t/official-samsung-odin-v4-1-2-1-dc05e3ea-for-linux.4453423/
- reboot into download mode:
  - pwr+vol-up+vol-down
  - vol-up
- `odin -l` to see device
- Note, THERE IS NO CP file! You have wifi-only device, so there
  is no "Cell Processor" file!


```
./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_2VTTH.tar \
-s HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/010
```



Um... fail?

```
(3116) ➜  Androidz ./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_2VTTH.tar \
-s HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/010
Check file : BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5
Check file : magisk_patched-26400_2VTTH.tar
Check file : HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5
meta-data/download-list.txt
/dev/bus/usb/003/010
Setup Connection
initializeConnection
Receive PIT Info
success getpit
Upload Binaries
abl.elf.lz4
xbl.elf.lz4
xbl_config.elf.lz4
tz.mbn.lz4
hyp.mbn.lz4
devcfg.mbn.lz4
aop.mbn.lz4
qupv3fw.elf.lz4
NON-HLOS.bin.lz4
dspso.bin.lz4
storsec.mbn.lz4
sec.elf.lz4
bksecapp.mbn.lz4
tz_iccc.mbn.lz4
tz_hdm.mbn.lz4
apdp.mbn.lz4
uefi_sec.mbn.lz4
vbmeta.img.lz4
vaultkeeper.mbn.lz4
recovery.img.lz4
super.img.lz4
dtbo.img.lz4
vbmeta.img
boot.img
Fail request receive -5
FAIL! (Auth)
Fail uploadBinaries
```

- Bootloader was locked (`OEM LOCK: ON (U)`).

- Unlocked with long press up on download
- Now `OEM LOCK: OFF (U)`, good to go?

```
./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_2VTTH.tar \
-s HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/020
```


```
./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_2VTTH.tar \
-s HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/025
```

 **SUCCESS**

 ```
 (3116) ➜  Androidz ./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_2VTTH.tar \
-s HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/025
Check file : BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5
Check file : magisk_patched-26400_2VTTH.tar
Check file : HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5
meta-data/download-list.txt
/dev/bus/usb/003/025
Setup Connection
initializeConnection
Receive PIT Info
success getpit
Upload Binaries
abl.elf.lz4
xbl.elf.lz4
xbl_config.elf.lz4
tz.mbn.lz4
hyp.mbn.lz4
devcfg.mbn.lz4
aop.mbn.lz4
qupv3fw.elf.lz4
NON-HLOS.bin.lz4
dspso.bin.lz4
storsec.mbn.lz4
sec.elf.lz4
bksecapp.mbn.lz4
tz_iccc.mbn.lz4
tz_hdm.mbn.lz4
apdp.mbn.lz4
uefi_sec.mbn.lz4
vbmeta.img.lz4
vaultkeeper.mbn.lz4
recovery.img.lz4
super.img.lz4
dtbo.img.lz4
vbmeta.img
boot.img
cache.img.lz4
prism.img.lz4
optics.img.lz4
Close Connection
```

...maybe

It's in a boot loop...

Yep...
Needed to factory reset.

**UNSUCCESSFUL**!

https://xdaforums.com/t/bootloop-android-10-enablefilecrypto_failed-boot-issue-after-recent-update.4063089/


---

Okay, factory-reset'd, which fixed the bootloop.

- Magisk patch again
- This time, `adb pull` the patched AP, rather than MTP transfer
- Let's try again

- ALSO, you INCORRECTLY selected `HOME_CSC` for the `-s` `CSC` option

```
./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_X5pFs.tar \
-s CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/037
```


```
(3116) ➜  Androidz ./odin4 \
-b BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-a magisk_patched-26400_X5pFs.tar \
-s HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5 \
-d /dev/bus/usb/003/042
Check file : BL_T970XXU4DWH3_T970XXU4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5
Check file : magisk_patched-26400_X5pFs.tar
Check file : HOME_CSC_OXM_T970OXM4DWH3_MQB69282250_REV00_user_low_ship_MULTI_CERT.tar.md5
meta-data/download-list.txt
/dev/bus/usb/003/042
Setup Connection
initializeConnection
Receive PIT Info
success getpit
Upload Binaries
abl.elf.lz4
xbl.elf.lz4
xbl_config.elf.lz4
tz.mbn.lz4
hyp.mbn.lz4
devcfg.mbn.lz4
aop.mbn.lz4
qupv3fw.elf.lz4
NON-HLOS.bin.lz4
dspso.bin.lz4
storsec.mbn.lz4
sec.elf.lz4
bksecapp.mbn.lz4
tz_iccc.mbn.lz4
tz_hdm.mbn.lz4
apdp.mbn.lz4
uefi_sec.mbn.lz4
vbmeta.img.lz4
vaultkeeper.mbn.lz4
recovery.img.lz4
super.img.lz4
dtbo.img.lz4
vbmeta.img
boot.img
cache.img.lz4
prism.img.lz4
optics.img.lz4
Close Connection

```

IT WORKED! (?)
