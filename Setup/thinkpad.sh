#=============================================================================#
#                                                                             #
#   88888888888 888      d8b          888      8888888b.                888   #
#       888     888      Y8P          888      888   Y88b               888   #
#       888     888                   888      888    888               888   #
#       888     88888b.  888 88888b.  888  888 888   d88P  8888b.   .d88888   #
#       888     888 "88b 888 888 "88b 888 .88P 8888888P"      "88b d88" 888   #
#       888     888  888 888 888  888 888888K  888        .d888888 888  888   #
#       888     888  888 888 888  888 888 "88b 888        888  888 Y88b 888   #
#       888     888  888 888 888  888 888  888 888        "Y888888  "Y88888   #
#                                                                             #
#=============================================================================#

# Power management.
tlp



# Trackpoint sensitivity on thinkpads
# ===================================

## FOR OLDER VERSIONS UBUNTU
# First, try out different values, 0-220, of settings
echo 80 |  sudo tee /sys/devices/platform/i8042/serio1/serio2/sensitivity
echo 100 | sudo tee /sys/devices/platform/i8042/serio1/serio2/speed

# After finding value you like, make new udev rule for sys start
sudo vi /etc/udev/rules.d/trackpoint.rules

# add this line
SUBSYSTEM=="serio", DRIVERS=="psmouse", DEVPATH=="/sys/devices/platform/i8042/serio1/serio2", ATTR{sensitivity}="80", ATTR{speed}="180"

# Reboot, or run commands:
sudo udevadm control --reload-rules
sudo udevadm trigger


## WHAT ACTUALLY WORKS CURRENT

# First, figure out which device id
xinput | grep -i trackpoint
⎜   ↳ TPPS/2 ALPS TrackPoint                    id=16   [slave  pointer  (2)]


# See what properties associated
xinput --list-props 12
#    ...
#    libinput Accel Speed (315): 0.000000
#    libinput Accel Speed Default (316): 0.000000
#    ...

# Experiment with speed setting, eg:
xinput --set-prop 12 'libinput Accel Speed' -0.7

# Once you're satisfied, add a startup application with that command

#=============================================================================#
#                                                                             #
#  ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▘  ▀▀▀▀▀▀    ▀▀▀▀▀▀                                       #
#  ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▀▀ ▀▀▀▀▀▀▘  ▝▀▀▀▀▀▀                                       #
#    ▀▀▀     ▀▀▀   ▀▀▀   ▀▀▀▀▀  ▀▀▀▀▀                                         #
#    ▀▀▀     ▀▀▀▀▀▀▀▀    ▀▀▀▝▀▘▝▀▘▀▀▀                                         #
#    ▀▀▀     ▀▀▀▀▀▀▀▀    ▀▀▀ ▀▀▀▀ ▀▀▀                                         #
#    ▀▀▀     ▀▀▀   ▀▀▀   ▀▀▀ ▝▀▀▘ ▀▀▀                                         #
#  ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▀▀ ▀▀▀▀▀  ▀▀  ▀▀▀▀▀                                       #
#  ▀▀▀▀▀▀▀ ▀▀▀▀▀▀▀▀▀▘  ▀▀▀▀▀      ▀▀▀▀▀                                       #
#                ▄▄▄ ▄   ▄     ▄   ▄▄▖      ▄                                 #
#                 █  █▄▖ ▄ ▄▄▖ █ ▄ █ █ ▄▖ ▗▄█                                 #
#                 █  █ █ █ █ █ █▟▘ █▀▘▗▄█ █ █                                 #
#                 █  █ █ █ █ █ █▝▙ █  ▜▄█ ▜▄█                                 #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#                                                                             #
#=============================================================================#

                 .88888888:.
                88888888.88888.
              .8888888888888888.
              888888888888888888
              88' _`88'_  `88888
              88 88 88 88  88888
              88_88_::_88_:88888
              88:::,::,:::::8888
              88`:::::::::'`8888
             .88  `::::'    8:88.
            8888            `8:888.
          .8888'             `888888.
         .8888:..  .::.  ...:'8888888:.
        .8888.'     :'     `'::`88:88888
       .8888        '         `.888:8888.
      888:8         .           888:88888
    .888:88        .:           888:88888:
    8888888.       ::           88:888888
    `.::.888.      ::          .88888888
   .::::::.888.    ::         :::`8888'.:.
  ::::::::::.888   '         .::::::::::::
  ::::::::::::.8    '      .:8::::::::::::.
 .::::::::::::::.        .:888:::::::::::::
 :::::::::::::::88:.__..:88888:::::::::::'
  `'.:::::::::::88888888888.88:::::::::'
        `':::_:' -- '' -'-' `':_::::'`

------------------------------------------------
Thank you for visiting https://asciiart.website/
This ASCII pic can be found at
https://asciiart.website/index.php?art=logos%20and%20insignias/linux

                 .88888888:.
                88888888.88888.
              .8888888888888888.
              888888888888888888
              88' _`88'_  `88888
              88 88 88 88  88888
              88_88_::_88_:88888
              88:::,::,:::::8888
              88`:::::::::'`8888
             .88  `::::'    8:88.
            8888            `8:888.
          .8888'             `888888.
         .8888:..  .::.  ...:'8888888:.
        .8888.'     :'     `'::`88:88888
       .8888        '         `.888:8888.
      888:8         .           888:88888
    .888:88        .:           888:88888:
    8888888.       ::           88:888888
    `.::.888.      ::          .88888888
   .::::::.888.    ::         :::`8888'.:.
  ::::::::::.888   '         .::::::::::::
  ::::::::::::.8    '      .:8::::::::::::.
 .::::::::::::::.        .:888:::::::::::::
 :::::::::::::::88:.__..:88888:::::::::::'
  `'.:::::::::::88888888888.88:::::::::'
miK     `':::_:' -- '' -'-' `':_::::'`



                                .:xxxxxxxx:.
                             .xxxxxxxxxxxxxxxx.
                            :xxxxxxxxxxxxxxxxxxx:.
                           .xxxxxxxxxxxxxxxxxxxxxxx:
                          :xxxxxxxxxxxxxxxxxxxxxxxxx:
                          xxxxxxxxxxxxxxxxxxxxxxxxxxX:
                          xxx:::xxxxxxxx::::xxxxxxxxx:
                         .xx:   ::xxxxx:     :xxxxxxxx
                         :xx  x.  xxxx:  xx.  xxxxxxxx
                         :xx xxx  xxxx: xxxx  :xxxxxxx
                         'xx 'xx  xxxx:. xx'  xxxxxxxx
                          xx ::::::xx:::::.   xxxxxxxx
                          xx:::::.::::.:::::::xxxxxxxx
                          :x'::::'::::':::::':xxxxxxxxx.
                          :xx.::::::::::::'   xxxxxxxxxx
                          :xx: '::::::::'     :xxxxxxxxxx.
                         .xx     '::::'        'xxxxxxxxxx.
                       .xxxx                     'xxxxxxxxx.
                     .xxxx                         'xxxxxxxxx.
                   .xxxxx:                          xxxxxxxxxx.
                  .xxxxx:'                          xxxxxxxxxxx.
                 .xxxxxx:::.           .       ..:::_xxxxxxxxxxx:.
                .xxxxxxx''      ':::''            ''::xxxxxxxxxxxx.
                xxxxxx            :                  '::xxxxxxxxxxxx
               :xxxx:'            :                    'xxxxxxxxxxxx:
              .xxxxx              :                     ::xxxxxxxxxxxx
              xxxx:'                                    ::xxxxxxxxxxxx
              xxxx               .                      ::xxxxxxxxxxxx.
          .:xxxxxx               :                      ::xxxxxxxxxxxx::
          xxxxxxxx               :                      ::xxxxxxxxxxxxx:
          xxxxxxxx               :                      ::xxxxxxxxxxxxx:
          ':xxxxxx               '                      ::xxxxxxxxxxxx:'
            .:. xx:.                                   .:xxxxxxxxxxxxx'
          ::::::.'xx:.            :                  .:: xxxxxxxxxxx':
  .:::::::::::::::.'xxxx.                            ::::'xxxxxxxx':::.
  ::::::::::::::::::.'xxxxx                          :::::.'.xx.'::::::.
  ::::::::::::::::::::.'xxxx:.                       :::::::.'':::::::::
  ':::::::::::::::::::::.'xx:'                     .'::::::::::::::::::::..
    :::::::::::::::::::::.'xx                    .:: :::::::::::::::::::::::
  .:::::::::::::::::::::::. xx               .::xxxx :::::::::::::::::::::::
  :::::::::::::::::::::::::.'xxx..        .::xxxxxxx ::::::::::::::::::::'
  '::::::::::::::::::::::::: xxxxxxxxxxxxxxxxxxxxxxx :::::::::::::::::'
    '::::::::::::::::::::::: xxxxxxxxxxxxxxxxxxxxxxx :::::::::::::::'
        ':::::::::::::::::::_xxxxxx::'''::xxxxxxxxxx '::::::::::::'
             '':.::::::::::'                        `._'::::::''

------------------------------------------------
Thank you for visiting https://asciiart.website/
This ASCII pic can be found at
https://asciiart.website/index.php?art=logos%20and%20insignias/linux


                                                                 #####
                                                                #######
                   #                                            ##O#O##
  ######          ###                                           #VVVVV#
    ##             #                                          ##  VVV  ##
    ##         ###    ### ####   ###    ###  ##### #####     #          ##
    ##        #  ##    ###    ##  ##     ##    ##   ##      #            ##
    ##       #   ##    ##     ##  ##     ##      ###        #            ###
    ##          ###    ##     ##  ##     ##      ###       QQ#           ##Q
    ##       # ###     ##     ##  ##     ##     ## ##    QQQQQQ#       #QQQQQQ
    ##      ## ### #   ##     ##  ###   ###    ##   ##   QQQQQQQ#     #QQQQQQQ
  ############  ###   ####   ####   #### ### ##### #####   QQQQQ#######QQQQQ

unknown
