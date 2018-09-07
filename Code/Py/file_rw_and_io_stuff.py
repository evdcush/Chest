import os, sys
import glob

#==============================================================================
# Get list of files
#==============================================================================
# SEE MORE METHODS: https://stackoverflow.com/a/41447012/6880404

'''
# Current dir contents:
- work.txt
- template.rst
- notes_on_thing.rst
- notes_on_thing2.rst
- cheat_sheet.rst
- read_script.py
- file_rw_and_io_stuff.py
'''

# Everything in that dir
# ========================================
files_from_os = os.listdir()
'''
print(files_from_os)
# just every file a that dir
['work.txt', 'template.rst', 'notes_on_thing.rst', 'notes_on_thing2.rst',
 'cheat_sheet.rst', 'read_script.py', 'file_rw_and_io_stuff.py',]
'''

# Matching files of same type or something in common
# ========================================
rst_files = []
for file in glob.glob("*.rst"):
    rst_files.append(file)
# Or simply:
rst_files = [f for f in glob.glob("*.rst")]
'''
print(rst_files)
# Now only files with .rst extension:
['template.rst', 'notes_on_thing.rst', 'notes_on_thing2.rst', 'cheat_sheet.rst',]
'''

