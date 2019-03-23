#!/usr/bin/env python

import os
import sys
import traceback
import subprocess
from pprint import pprint

import pyperclip
from slugify import slugify
from pylibgen import Library


downloads_dir = os.environ['HOME'] + '/Downloads'


def fname_book(book):
    title = slugify(book.title)
    fname = f'{downloads_dir}/{title}.{book.extension}'
    return fname

def get_book(book):
    fname = fname_book(book)
    subprocess.run(f'wget -O {fname} {book.get_url()}', shell=True)

lgen = Library()

def search(isbn):
    ids = lgen.search(isbn)
    if len(ids) == 0:
        print(f'No results for id {isbn} !')
        return
    books = list(lgen.lookup(ids))
    for i, b in enumerate(books):
        if b.extension != 'pdf':
            continue
        b.filesize = f'{int(b.filesize) / 1024**2 :.2f}mb'
        print(f'BOOK {i}')
        pprint(vars(b))
        print('\n')
    bidx = input('Download book (i/N): ')
    if bidx:
        #assert bidx.isdigit() # its just u
        book = books[int(bidx)]
        get_book(book)
    return


def main():
    # isbn can be input via stdin OR from clipboard
    isbn = sys.argv[-1] if len(sys.argv) > 1 else pyperclip.paste()
    search(isbn)
    return 0

if __name__ == '__main__':
    try:
        ret = main()
    except:
        traceback.print_exc()
    sys.exit(ret)


