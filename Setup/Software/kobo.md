## Bypass Kobo registration/account after reset
So after resetting kobo aura, it REQUIRED that you sign in with some account. You cannot use the kobo you bought without signing up.


*Nah:*
```
# Connect your device to your computer
cd /media/$USER/KOBOeReader/.kobo

# Insert fake user
sqlite3 KoboReader.sqlite
delete from user;
insert into user values ('','','','','','','');
.quit

# Eject and unplug, and it should go straight to home screen!
```

#### References:
https://www.mobileread.com/forums/showthread.php?t=171664
https://wiki.mobileread.com/wiki/Kobo_Touch_Hacking#Fake_registration
