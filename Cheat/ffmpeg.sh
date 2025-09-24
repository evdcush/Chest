# convert audio file format from XXX to YYY losslessly
# (ie, codec copy)

##  opus to wav ()
ffmpeg -i ayy.opus -c:a pcm_s16le ayy.wav

## m4a to wav
ffmpeg -i foo.m4a -c:a pcm_s16le foo.wav


#===============================  compression  ===============================#

# Compress a 4K60HZ video to 1080p-30HZ video (lossy)
##  PURPOSE: you've got a big-ass vid file. You want smol one.
##  args:
##    - vf 'scale=1920:-2,fps30': "resize resolution: width 3840 to 1920, height autoscale, to 30hz"
##    -c:v libx265 : use video codec h265
##    -crf : default 23, lower the better, higher the lossier (28 is often used for less-noticeable compression)
##    -preset slow : dun remember, makes ffmpeg process slower but "better" i guess
##    -c:a aac : use aac audio codec
##    -b:a 128k : use 128k bitrate for audio
ffmpeg -i \
my_big_4k60hz_vid.mp4 \
-vf "scale=1920:-2,fps=30" \
-c:v libx265 -crf 30 -preset slow \
-c:a aac -b:a 128k \
my_big_vid_compressed_to_1080p30hz.mp4


#=============================================================================#
#                                     DUMP                                    #
#=============================================================================#
# the amount of ffmpeg shell history you have is insane
# maybe instead of always:
#   ctrl+r -> fzf -> 'ffmpeg' -> {browse for similar looking cmd} ->
#   copy-pa -> "what do these flags mean again?" -> tldr ffmpeg (not there) ->
#   ffmpeg --help (oh god oh no what not all this) ->
#   cheat ffmpeg (if installed) -> close enough ->
#   ask ai bot -> finally get cmd (but don't learn the flag--again)
#              -> get cmd doesn't work bc i didn't ask the question or specify
#                 my use-case exactly because i dont know what i dont know
#                 end up musing about ai brittleness
#   search online -> remember that good god, like regex, there is so much
#                    here, easily find the command ->
#   great success! ->
#   don't bother to write somewhere what the command does or the flags mean ->
#   ctrl+r -> fzf -> 'ffmpeg' -> {browse for similar looking cmd} -> ....
#
#   so, maybe, start saving your snippets here
#
#   HERE'S THE DUMP! (from 2501P14sA500 local hist)
 1057  alias | grep ffmpeg
 1058  ffmpeg -ss 00:10:16 -to 00:11:40 -i Drag_Racing_the_Whipple_F-150_and_Rat_Rod_Supra_They_re_WAY_FASTER_Than_We_Thought.-qKbiUJcJ_s.webm -codec copy ratrod_rip.mp4
 1059  ffmpeg -ss 00:10:16 -to 00:11:40 -i Drag_Racing_the_Whipple_F-150_and_Rat_Rod_Supra_They_re_WAY_FASTER_Than_We_Thought.-qKbiUJcJ_s.webm ratrod_rip.mp4
 1062  ffmpeg -ss 00:10:16 -to 00:11:40 -i Drag\ Racing\ the\ Whipple\ F-150\ and\ Rat\ Rod\ Supra\!\ They\'re\ WAY\ FASTER\ Than\ We\ Thought\!\!\!\ \[-qKbiUJcJ_s\].mp4 ratrod-rip.mp4
 1063  ffmpeg -crf 28 -ss 00:10:16 -to 00:11:40 -i Drag\ Racing\ the\ Whipple\ F-150\ and\ Rat\ Rod\ Supra\!\ They\'re\ WAY\ FASTER\ Than\ We\ Thought\!\!\!\ \[-qKbiUJcJ_s\].mp4 ratrod-rip28.mp4
 1064  ffmpeg -ss 00:10:16 -to 00:11:40 -i Drag\ Racing\ the\ Whipple\ F-150\ and\ Rat\ Rod\ Supra\!\ They\'re\ WAY\ FASTER\ Than\ We\ Thought\!\!\!\ \[-qKbiUJcJ_s\].mp4 -crf 28 ratrod-rip28.mp4
 1065  ffmpeg -ss 00:10:16 -to 00:11:40 -i Drag\ Racing\ the\ Whipple\ F-150\ and\ Rat\ Rod\ Supra\!\ They\'re\ WAY\ FASTER\ Than\ We\ Thought\!\!\!\ \[-qKbiUJcJ_s\].mp4 -vcodec libx265 -crf 28 ratrod-rip28.mp4
 1066  ffmpeg -ss 00:10:16 -to 00:11:40 -i Drag\ Racing\ the\ Whipple\ F-150\ and\ Rat\ Rod\ Supra\!\ They\'re\ WAY\ FASTER\ Than\ We\ Thought\!\!\!\ \[-qKbiUJcJ_s\].mp4 -vcodec libx265 -crf 32 ratrod-rip32.mp4
 1069  ffmpeg -ss 00:01:52 -to 00:02:13 -i Jzx110\ Back\ Road\ Blasting\ 3\ \[_Ev2jLsdXiA\].mp4 chaser-rip-tsutsutsutsu.mp4
 1070  ffmpeg -ss 00:01:52 -to 00:02:13 -i Jzx110\ Back\ Road\ Blasting\ 3\ \[_Ev2jLsdXiA\].mp4 -vcodec libx265 -crf 28 chaser-rip-tsutsutsutsu.mp4
 4233  ffmpeg -i Geto\ Boys\ -\ Damn\ It\ Feels\ Good\ To\ Be\ A\ Gangsta.webm -to 01:08 -codec copy everything_cool_in_the_mind_of_a_gangsta.webm
 4753  ffmpeg -i amz-shipping-restrictions_video-demo_2025-04-08.mp4 -c copy -an amz-shipping-restrictions_video-demo_2025-04-08_noaudio.mp4
 4755  ffmpeg -i foo.mp4 -c:v libx264 -preset fast -crf 23 -an foo_no_audio.mp4
 4757  ffmpeg -i amz-shipping-restrictions_video-demo_2025-04-08.mp4 -c:v libx264 -preset fast -crf 25 -an amz-shipping-restrictions_video-demo_2025-04-08_noaudio.mp4
 6551  ffmpeg -f pulse -i default output.wav
 6956  ffmpeg -i 'call with da ma.m4a' -ac 1 -ar 16000 'call with da ma.flac'
 7315  ffmpeg -i '/home/evan/Media/one-third.mkv' \\n  -c:v libx265 -crf 28 \\n  -c:a copy \\n  -preset slow \\n  '/home/evan/Media/one-third-small.mkv'
 9992  ffmpeg -i 'call with da ma.m4a' -c:a pcm_s16le 'call with da ma.wav'
 9995  ffmpeg ffmpeg -i 'call with da ma.wav' -c:a libmp3lame -b:a 320k 'call with da ma.mp3'


#   HERE'S THE DUMP! (from swol-22-2xchad-cards local hist)
##  NB: some of these commands may be wrong
##      also, wtf.... this is absolutely not the complete history I have for ffmpeg cmds?
  673  ffmpeg -i 1.\ Roman\ Britain\ -\ The\ Work\ of\ Giants\ Crumbled.2ae378.mp3 -vn -acodec copy -ss 00:19:16 -to 00:20:25 'Fall of Civilizations - Roman Britain - The Work of Giants Crumbled.mp3'
  787  ffmpeg -ss 00:03:38 -to 00:06:31 -i Cleetus\ McFarland\ -\ We\ Turned\ The\ Rat\ Rod\ Supra\ ALL\ THE\ WAY\ UP\ and\ It\ Pulls\ So\ Hard....mkv -vcodec libx265 -crf 28 justbuilt2jzthings.mp4
  789  ffmpeg -ss 00:03:38 -to 00:06:31 -hwaccel cuda -hwaccel_output_format cuda -i Cleetus\ McFarland\ -\ We\ Turned\ The\ Rat\ Rod\ Supra\ ALL\ THE\ WAY\ UP\ and\ It\ Pulls\ So\ Hard....mkv -c:v hevc_nvenc -cq 32 justbuilt2jzstuff.mp4
  790  ffmpeg -ss 00:03:38 -to 00:06:31 -hwaccel cuda -hwaccel_output_format cuda -extra_hw_frames 16 -i Cleetus\ McFarland\ -\ We\ Turned\ The\ Rat\ Rod\ Supra\ ALL\ THE\ WAY\ UP\ and\ It\ Pulls\ So\ Hard....mkv -c:v hevc_nvenc -cq 32 foo.mp4
  908  ffmpeg -i 'andrej - lets build gpt from scratch.mkv' -vn karpathy_build_gpt_scratch.opus
  909  ffmpeg -i 'andrej - lets build gpt from scratch.mkv' -vn karpathy_build_gpt_scratch.mp3
 1272  ffmpeg -i scout-at-the-beach-2024-August.mp4 -vcodec h264 -acodec aac scout-at-the-beach-2024-August-enc.mp4
 2482  ffmpeg -i "concat:cap-42069-edit-1969-04-20-04_20_11_EDITED1.png|cap-42069-edit-1969-04-20-04_20_11-2_EDITED1.png" -filter_complex vstack combined.png
 4815  ffmpeg -i "srachk.mkv" -vn -acodec pcm_s16le output.wav
 4981  ffmpeg -i "ComputerVisionFoundation Videos - SSD-6D： Making RGB-Based 3D Detection and 6D Pose Estimation Great Again.YBwHZ8yOXfc.mkv" -vn -acodec pcm_s16le output.wav
 4984  ffmpeg -i "Shia LaBeouf 'Just Do It' Motivational Speech (Original Video by LaBeouf, Rönkkö & Turner).ZXsQAXx_ao0.mkv" -vn -acodec pcm_s16le shia_output.wav
 4988  ffmpeg -i king_motivational_speech4.wav -af "compand=attacks=0:decays=0:points=-80/-80|-40/-60|0/0, dynaudnorm" king_clean_shia.wav
 4989  ffmpeg -i king_motivational_speech4.wav \\n-af "agate=threshold=-45dB:ratio=10:attack=5ms:release=200ms" \\nking_gated.wav\n
 5298  for f in *.jpg; do ffmpeg -y -i "$f" -preset medium -qscale:v 23 "${f%.jpg}c.jpeg"; done
 5299  for f in *.jpg; do ffmpeg -y -i "$f" -map_metadata 0 -preset medium -qscale:v 23 "${f%.jpg}c.jpeg"; done
 5300  for f in *.jpg; do ffmpeg -y -i "$f" -map_metadata 0 -preset medium -qscale:v 23 "${f%.jpg}c.jpeg"; done\n
 5305  for f in *.jpg; do ffmpeg -y -i "$f" -map_metadata 0 -preset medium -qscale:v 23 "${f%.jpg}.jpeg"; done
 5602  ffmpeg -f pulse -i combined_source foo.wav
 5604  ffmpeg \\n  -f pulse -i alsa_output.usb-DSEA_A_S_EPOS_IMPACT_60_A001290220401692-00.analog-stereo.monitor \\n  -f pulse -i alsa_input.usb-DSEA_A_S_EPOS_IMPACT_60_A001290220401692-00.mono-fallback \\n  -filter_complex amix=inputs=2:duration=longest \\n  output.wav
 6655  ffmpeg -i 'Ghost 01 - did somebody call for an exterminator.opus' -f ffmetadata -
 6656  ffmpeg -i 'Ghost 01 - did somebody call for an exterminator.opus' -c copy -map_metadata -1 output.opus
 6657  ffmpeg -i 'Ghost 01 - did somebody call for an exterminator.opus' -c:a copy -map_metadata -1 output.opus
 6658  ffmpeg -f ffmetadata - -i output.opus
 6659  ffmpeg -i output.opus -f ffmetadata -
 6667  for file in *.opus; do\n    ffmpeg -i "$file" -c:a copy -map_metadata -1 "${file}"\ndone
 6669  for file in *.opus; do\n    ffmpeg -i "$file" -c:a copy -map_metadata -1 "clean_${file}"\ndone
 6671  ffmpeg -i 'Battlecruiser (StarCraft) (1) [Battlecruiser_(StarCraft)-1].opus' -c:a copy -map_metadata -1 'Battlecruiser 01 - battlecruiser operational.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (2) [Battlecruiser_(StarCraft)-2].opus' -c:a copy -map_metadata -1 'Battlecruiser 02 - all crews reporting.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (3) [Battlecruiser_(StarCraft)-3].opus' -c:a copy -map_metadata -1 'Battlecruiser 03 - receiving transmission.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (4) [Battlecruiser_(StarCraft)-4].opus' -c:a copy -map_metadata -1 'Battlecruiser 04 - good day commander.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (5) [Battlecruiser_(StarCraft)-5].opus' -c:a copy -map_metadata -1 'Battlecruiser 05 - alien frequencies open.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (6) [Battlecruiser_(StarCraft)-6].opus' -c:a copy -map_metadata -1 'Battlecruiser 06 - make it happen.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (7) [Battlecruiser_(StarCraft)-7].opus' -c:a copy -map_metadata -1 'Battlecruiser 07 - set a course.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (8) [Battlecruiser_(StarCraft)-8].opus' -c:a copy -map_metadata -1 'Battlecruiser 08 - take it slow.opus';\nffmpeg -i 'Battlecruiser (StarCraft) (9) [Battlecruiser_(StarCraft)-9].opus' -c:a copy -map_metadata -1 'Battlecruiser 09 - in the cage.opus';
 6673  ffmpeg -i 'Archon (StarCraft) (1) [Archon_(StarCraft)-1].opus' -c:a copy -map_metadata -1 'Archon 1 - the merging is complete.opus';\nffmpeg -i 'Archon (StarCraft) (2) [Archon_(StarCraft)-2].opus' -c:a copy -map_metadata -1 'Archon 2 - we burn.opus';\nffmpeg -i 'Archon (StarCraft) (3) [Archon_(StarCraft)-3].opus' -c:a copy -map_metadata -1 'Archon 3 - we need focus.opus';\nffmpeg -i 'Archon (StarCraft) (5) [Archon_(StarCraft)-5].opus' -c:a copy -map_metadata -1 'Archon 5 - power overwhelming.opus';\nffmpeg -i 'Archon (StarCraft) (6) [Archon_(StarCraft)-6].opus' -c:a copy -map_metadata -1 'Archon 6 - destroy.opus';\nffmpeg -i 'Archon (StarCraft) (7) [Archon_(StarCraft)-7].opus' -c:a copy -map_metadata -1 'Archon 7 - annihilate.opus';\nffmpeg -i 'Archon (StarCraft) (8) [Archon_(StarCraft)-8].opus' -c:a copy -map_metadata -1 'Archon 8 - obliterate.opus';\nffmpeg -i 'Archon (StarCraft) (9) [Archon_(StarCraft)-9].opus' -c:a copy -map_metadata -1 'Archon 9 - eradicate.opus';
 6678  ffmpeg -i 'Firebat (StarCraft) (1) [Firebat_(StarCraft)-1].opus' -c:a copy -map_metadata -1 'Firebat 01 - need a light.opus';\nffmpeg -i 'Firebat (StarCraft) (2) [Firebat_(StarCraft)-2].opus' -c:a copy -map_metadata -1 'Firebat 02 - fire it up.opus';\nffmpeg -i 'Firebat (StarCraft) (3) [Firebat_(StarCraft)-3].opus' -c:a copy -map_metadata -1 'Firebat 03 - yes.opus';\nffmpeg -i 'Firebat (StarCraft) (4) [Firebat_(StarCraft)-4].opus' -c:a copy -map_metadata -1 'Firebat 04 - you got my attention.opus';\nffmpeg -i 'Firebat (StarCraft) (5) [Firebat_(StarCraft)-5].opus' -c:a copy -map_metadata -1 'Firebat 05 - want to turn up the heat.opus';\nffmpeg -i 'Firebat (StarCraft) (6) [Firebat_(StarCraft)-6].opus' -c:a copy -map_metadata -1 'Firebat 06 - naturally.opus';\nffmpeg -i 'Firebat (StarCraft) (7) [Firebat_(StarCraft)-7].opus' -c:a copy -map_metadata -1 'Firebat 07 - slammin.opus';\nffmpeg -i 'Firebat (StarCraft) (8) [Firebat_(StarCraft)-8].opus' -c:a copy -map_metadata -1 'Firebat 08 - youve got it.opus';\nffmpeg -i 'Firebat (StarCraft) (9) [Firebat_(StarCraft)-9].opus' -c:a copy -map_metadata -1 'Firebat 09 - lets burn.opus';
 6683  ffmpeg -i 'Zealot (StarCraft) (1) [Zealot_(StarCraft)-1].opus' -c:a copy -map_metadata -1 'Zealot 01 - my life for aiur.opus';\nffmpeg -i 'Zealot (StarCraft) (2) [Zealot_(StarCraft)-2].opus' -c:a copy -map_metadata -1 'Zealot 02 - what now calls.opus';\nffmpeg -i 'Zealot (StarCraft) (3) [Zealot_(StarCraft)-3].opus' -c:a copy -map_metadata -1 'Zealot 03 - issah tsu.opus';\nffmpeg -i 'Zealot (StarCraft) (4) [Zealot_(StarCraft)-4].opus' -c:a copy -map_metadata -1 'Zealot 04 - i long for combat.opus';\nffmpeg -i 'Zealot (StarCraft) (5) [Zealot_(StarCraft)-5].opus' -c:a copy -map_metadata -1 'Zealot 05 - jee house.opus';\nffmpeg -i 'Zealot (StarCraft) (6) [Zealot_(StarCraft)-6].opus' -c:a copy -map_metadata -1 'Zealot 06 - gau gurah.opus';\nffmpeg -i 'Zealot (StarCraft) (7) [Zealot_(StarCraft)-7].opus' -c:a copy -map_metadata -1 'Zealot 07 - thus i serve.opus';\nffmpeg -i 'Zealot (StarCraft) (8) [Zealot_(StarCraft)-8].opus' -c:a copy -map_metadata -1 'Zealot 08 - honor guide me.opus';\nffmpeg -i 'Zealot (StarCraft) (9) [Zealot_(StarCraft)-9].opus' -c:a copy -map_metadata -1 'Zealot 09 - for adun.opus';
 6686  ffmpeg -i 'Valkyrie (1) [Valkyrie#StarCraft-1].opus'-c:a copy -map_metadata -1 'Valkyrie 01 - valykrie prepared.opus';\nffmpeg -i 'Valkyrie (2) [Valkyrie#StarCraft-2].opus'-c:a copy -map_metadata -1 'Valkyrie 02 - need something destroyed.opus';\nffmpeg -i 'Valkyrie (3) [Valkyrie#StarCraft-3].opus'-c:a copy -map_metadata -1 'Valkyrie 03 - i am eager to help.opus';\nffmpeg -i 'Valkyrie (4) [Valkyrie#StarCraft-4].opus'-c:a copy -map_metadata -1 'Valkyrie 04 - dont keep me waiting.opus';\nffmpeg -i 'Valkyrie (5) [Valkyrie#StarCraft-5].opus'-c:a copy -map_metadata -1 'Valkyrie 05 - achtung.opus';\nffmpeg -i 'Valkyrie (6) [Valkyrie#StarCraft-6].opus'-c:a copy -map_metadata -1 'Valkyrie 06 - of course mein herr.opus';\nffmpeg -i 'Valkyrie (7) [Valkyrie#StarCraft-7].opus'-c:a copy -map_metadata -1 'Valkyrie 07 - perfect.opus';\nffmpeg -i 'Valkyrie (8) [Valkyrie#StarCraft-8].opus'-c:a copy -map_metadata -1 'Valkyrie 08 - its show time.opus';\nffmpeg -i 'Valkyrie (9) [Valkyrie#StarCraft-9].opus'-c:a copy -map_metadata -1 'Valkyrie 09 - jawohl.opus';
 6687  ffmpeg -i 'Valkyrie (1) [Valkyrie#StarCraft-1].opus' -c:a copy -map_metadata -1 'Valkyrie 01 - valykrie prepared.opus';\nffmpeg -i 'Valkyrie (2) [Valkyrie#StarCraft-2].opus' -c:a copy -map_metadata -1 'Valkyrie 02 - need something destroyed.opus';\nffmpeg -i 'Valkyrie (3) [Valkyrie#StarCraft-3].opus' -c:a copy -map_metadata -1 'Valkyrie 03 - i am eager to help.opus';\nffmpeg -i 'Valkyrie (4) [Valkyrie#StarCraft-4].opus' -c:a copy -map_metadata -1 'Valkyrie 04 - dont keep me waiting.opus';\nffmpeg -i 'Valkyrie (5) [Valkyrie#StarCraft-5].opus' -c:a copy -map_metadata -1 'Valkyrie 05 - achtung.opus';\nffmpeg -i 'Valkyrie (6) [Valkyrie#StarCraft-6].opus' -c:a copy -map_metadata -1 'Valkyrie 06 - of course mein herr.opus';\nffmpeg -i 'Valkyrie (7) [Valkyrie#StarCraft-7].opus' -c:a copy -map_metadata -1 'Valkyrie 07 - perfect.opus';\nffmpeg -i 'Valkyrie (8) [Valkyrie#StarCraft-8].opus' -c:a copy -map_metadata -1 'Valkyrie 08 - its show time.opus';\nffmpeg -i 'Valkyrie (9) [Valkyrie#StarCraft-9].opus' -c:a copy -map_metadata -1 'Valkyrie 09 - jawohl.opus';
 6690  ffmpeg -i 'Siege tank (StarCraft) (1) [Siege_tank_(StarCraft)-1].opus' -c:a copy -map_metadata -1 'Siege Tank 01 - ready to roll out.opus'\nffmpeg -i 'Siege tank (StarCraft) (2) [Siege_tank_(StarCraft)-2].opus' -c:a copy -map_metadata -1 'Siege Tank 02 - yes sir.opus'\nffmpeg -i 'Siege tank (StarCraft) (3) [Siege_tank_(StarCraft)-3].opus' -c:a copy -map_metadata -1 'Siege Tank 03 - destination.opus'\nffmpeg -i 'Siege tank (StarCraft) (4) [Siege_tank_(StarCraft)-4].opus' -c:a copy -map_metadata -1 'Siege Tank 04 - identify target.opus'\nffmpeg -i 'Siege tank (StarCraft) (5) [Siege_tank_(StarCraft)-5].opus' -c:a copy -map_metadata -1 'Siege Tank 05 - orders sir.opus'\nffmpeg -i 'Siege tank (StarCraft) (6) [Siege_tank_(StarCraft)-6].opus' -c:a copy -map_metadata -1 'Siege Tank 06 - move it.opus'\nffmpeg -i 'Siege tank (StarCraft) (7) [Siege_tank_(StarCraft)-7].opus' -c:a copy -map_metadata -1 'Siege Tank 07 - pro ceedin.opus'\nffmpeg -i 'Siege tank (StarCraft) (8) [Siege_tank_(StarCraft)-8].opus' -c:a copy -map_metadata -1 'Siege Tank 08 - dee lighted to sir.opus'\nffmpeg -i 'Siege tank (StarCraft) (9) [Siege_tank_(StarCraft)-9].opus' -c:a copy -map_metadata -1 'Siege Tank 09 - ab so lutely.opus'
 6706  ffmpeg -i 'Battlecruiser 04 - good day commander.opus' 'Battlecruiser 04 - good day commander.wav'
 7244  convert "cap-42069-edit-1969-04-20-04_20_11_EDITED1.png" "cap-42069-edit-1969-04-20-04_20_11-2_EDITED1.png" -append "./combined.png"ffmpeg -i "concat:cap-42069-edit-1969-04-20-04_20_11_EDITED1.png|cap-42069-edit-1969-04-20-04_20_11-2_EDITED1.png" -filter_complex vstack combined.png
 8921  ffmpeg -i 'Sam Matla - The Complexity Trap.mkv' -vn -c:a copy sam_complexity_trap3.opus
 9087  ffmpeg -i 'Fireship - Raspberry Pi Explained in 100 Seconds.eZ74x6dVYes.mkv' -vn -c:a copy fireship_rpi.opus
 9716  ffmpeg -i koe_ivaylo_solo_v1_00003_CLEANEST-SO-FAR.flac -c copy eevylo_g5_bigD_playa.wav
 9717  ffmpeg -i koe_ivaylo_solo_v1_00003_CLEANEST-SO-FAR.flac  eevylo_g5_bigD_playa.mp3
 9827  ffmpeg -i "concat:$(printf '%s|' Ghost 0*.opus | sed 's/|$//')" -c copy Ghost.opus
 9828  ffmpeg -f concat -safe 0 -i <(for f in Ghost 0*.opus; do echo "file '$f'"; done) -c copy Ghost.opus
 9829  ffmpeg -f concat -safe 0 -i "Ghost 0*.opus" -c copy Ghost.opus
 9834  ffmpeg -f concat -safe 0 -i ghost_files.txt -c copy ghost.opus
 9836  ffmpeg -f concat -safe 0 -i bc_files.txt -c copy battlecruiser.opus
 9841  ffmpeg -f concat -i bc_files.txt -c copy battlecruiser2.opus
10079  man ffmpeg | cat | grep --context 5 -i preset
