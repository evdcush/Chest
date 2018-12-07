"""
# Acknowledgement
# ===============
script adapted from mazen160/GithubCloner
"""
import os
import git
import yaml
import json
import time
import requests
import threading
import datetime as dt
from pprint import pprint

# Constants
HOME = os.environ["HOME"]
RESOURCE_PATH = f'{HOME}/Chest/Resources'
DEST_PATH = f'{HOME}/Projects/Hoard/Clones'

with open(f'{HOME}/.gitconfig') as gc:
    USERNAME = [s for s in gc.readlines() if '@' in s][0].split(' = ')[1].split('@')[0]


class Cloner:
    API = ("https://api.github.com/{usr_type}/{target}/"
           "repos?per_page=40000000&page={{pg}}")
    headers = {'Accept': '*/*'}
    timeout = 3
    def __init__(self, clone_path=DEST_PATH, username=None, token=None):
        self.clone_path = clone_path
        self.username = username
        self.token = token
        if not os.path.exists(clone_path): os.mkdirs(clone_path)

    def check_target_response(self, response):
        """ Check for error in the response. """
        try:
            msg = response["message"]
            if 'exceeded' in msg:
                print('Github API rate limit exceeded')
                return 1
            elif msg == 'Not Found':
                print('Target not found')
                return 2
        except TypeError:
            pass
        return 0


    def get_urls_from_target(self, target_user, usr_type):
        """ # Retrieves a list of repos from target_user

        Params
        ------
        target_user : str
            github user from which repos retrieved
        usr_type : str, in ['org', 'user']
            whether target_user is organization or user

        Returns
        -------
        URLs : list(str)
            urls of user's repos
        """
        print(f'\nRetrieving repos from: {target_user}\n')
        api_base = self.API.format(usr_type=usr_type, target_user=target_user)
        URLs = []
        current_page = 1
        while (len(resp) != 0 or current_page == 1):
            #==== format api call
            api = api_base.format(pg=current_page)
            req_kwargs = dict(headers=self.HEADERS, timeout=self.TIMEOUT)
            if self.token is None:
                req_kwargs['auth'] = (self.username, self.token)

            #==== api get
            resp = json.loads(requests.get(api, **req_kwargs).text)
            if check_response(resp) != 0: return []

            #==== parse response
            for r in resp:
                URLs.append(r['git_url'])
            current_page += 1
        return URLs


    def clone_repo(self, URL):
        print(f'clone_repo({URL})')
        # Format URL
        # ==========
        URL = URL.replace("git://", "https://")
        replacement = "https://"
        #---- replace with username & token if avail
        if self.token is not None:
            replacement = f'https://{self.username}:{self.token}@'
        URL = URL.replace('git://', replacement)

        # Format repo clone path
        # ======================
        rpath = '{}_{}'.format(*URL.split('/')[-2:])
        if rpath[-4:] == '.git': rpath = rpath[:-4]
        if '@' in repo_path:
            idx = repo_path.index('@') + 1
            repo_path = repo_path.replace(repo_path[:idx], "")
        write_path = f'{self.clone_path}/{repo_path}'

        # Clone repo
        # ========
        try:
            with threading.Lock():
                print(write_path)
            if os.path.exists(write_path):
                git.Repo(write_path).remote().pull()
            else:
                git.Repo.clone_from(URL, repo_path, depth=1)
        except:
            print(f'\n{"="*80}\n\nError in cloning {URL}\n\n{"="*80}\n')


    def batch_clone(self, URLs, threads_limit=5):
        threads_state = []
        while URLs:
            if (threading.active_count() < (threads_limit + 1)):
                # thread
                t = threading.Thread(target=clone_repo, args=URLs.pop())
                t.daemon = True
                t.start()
            else:
                time.sleep(0.5)
                threads_state.append(t)
        for _ in threads_state:
            _.join()

#==============================================================================


# Interfacing code
# ================

#==== Loaders
yml_ld = lambda fpath: with open(f'{fpath}.yml') as fp: yaml.load(fp)

# Read file
def read_conf(fname, loader=yml_ld):
    file_path = f'{RESOURCE_PATH}/{fname}'
    return loader(file_path)




################## STOPPED HERE, UPDATE LATER
##########################################################
def get_urls(txt=repo_inbox, archive=True, username=None, token=None):
    with open(txt, 'r+') as txt_urls:
        url_list = txt_urls.read().split('\n')[:-1]
        urls = list(url_list)
        if archive:
            archive_urls(url_list, KEYS[txt])
            # clear inbox
            #txt_urls.truncate(0) # need '0' when using r+
        if txt in [org_inbox, user_inbox]:
            urls = []
            uo_get_func = fromOrg if txt == org_inbox else from_user
            for u in url_list:
                #code.interact(local=dict(globals(), **locals())) # DEBUGGING-use
                urls.extend(uo_get_func(u, username=username, token=token))
    return urls


urls = get_urls(org_inbox, username=USERNAME, token=TOKEN)
urls.extend(get_urls(user_inbox, username=USERNAME, token=TOKEN))
urls.extend(get_urls(repo_inbox, username=USERNAME, token=TOKEN))
#pprint(urls)

#urls = get_urls_from_file()
#pprint(urls)

cloneBulkRepos(urls, clone_path, threads_limit=8, username=USERNAME, token=TOKEN)

# URL origin
# ==========

#==== target_user

#==== Org

#==== txt
