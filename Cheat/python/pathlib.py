from pathlib import Path

>>> Path.home()
PosixPath('/home/evan')

>>> p = Path('configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py')

>>> p
PosixPath('configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py')

>>> p.absolute()
PosixPath('/home/evan/Projects/Gigachad/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py')

>>> p.absolute().as_uri()
'file:///home/evan/Projects/Gigachad/configs/mm_grounding_dino/grounding_dino_swin-t_pretrain_obj365.py'

>>> p.parts
('configs', 'mm_grounding_dino', 'grounding_dino_swin-t_pretrain_obj365.py')

>>> p.parent
PosixPath('configs/mm_grounding_dino')

>>> p.name
'grounding_dino_swin-t_pretrain_obj365.py'

>>> p.stem
'grounding_dino_swin-t_pretrain_obj365'

>>> p.suffix
'.py'

>>> p.is_file()
True

# Other handy stuffs:
'''
p.cwd()
p.is_dir()
p.is_link()
p.link_to()
p.open()
p.readtext()
'''
