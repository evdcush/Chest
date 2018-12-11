import os
import yaml
import subprocess
from pprint import pprint

# Paths
# =====
HOME = os.environ['HOME']
R_path = f'{HOME}/Chest/Resources/{{}}_hoard.yml'
W_path = f'{HOME}/Projects/Hoard/Archive'
if not os.path.exists(W_path): os.mkdirs(W_path)

# File RW helpers
# ===============
def yml_read(fname):
    with open(fname) as file:
        return yaml.load(file)

def yml_write(fname, obj):
    with open(fname, 'w') as file:
        yaml.dump(obj, file, default_flow_style=False)

# Process hoard inbox and outbox
# ==============================
archive = yml_read(R_path.format('archive'))
inbox   = yml_read(R_path.format('inbox'))

# Only get unique repos; (clobbered users & orgs OKAY)
repos = set(inbox['repos']).difference(set(archive['repos']))

# Get repos
# =========
def get_archive(repo_url):
    url = repo_url.strip('/')
    usr, repo = url.split('/')[-2:]
    save_path = f'{W_path}/{usr}_{repo}.zip'
    subprocess.run(f'wget {url}/archive/master.zip -O {save_path}', shell=True)

def get_all_repos(repos):
    # in: list of url (str)
    archived_urls_file = open('repos_archived.txt', 'x')
    while repos:
        repo_url = repos.pop()
        get_archive(repo_url)

        # Update archive and urls file
        archive['repos'].append(repo_url)
        archived_urls_file.write(repo_url + '\n')

    # Close archives
    archived_urls_file.close()
    yml_write(R_PATH.format('archive'), archive)

    # Update inbox
    inbox['repos'] = [None,]
    yml_write(R_PATH.format('inbox'), inbox)

if __name__ == '__main__':
    get_all_repos(repos)


# https://github.com/Mooophy/Cpp-Primer
# https://github.com/Mooophy/Cpp-Primer/archive/master.zip

# USERS & ORGS HOARDING:  DO NOT WORRY ABOUT RIGHT NOW
#  just do repos, then fig out how to parse users n orgs
###############################################################################
# https://github.com/Mooophy?tab=repositories
# https://api.github.com/users/Mooophy/repos
#"https://api.github.com/{usr_type}/{target}/repos?per_page=40000000&page={{pg}}"
#"https://api.github.com/users/Mooophy/repos?per_page=40000000&page=1
#====> Sample response below
#  it seems you want to make sure the obj does not have something
#  from the parent/wrapper obj. You want the "html_url" val for the
#  url to wget

""" # Yes, those are the actual repo names (159233, 159201, etc.)
[{
  "id": 28810282,
  "node_id": "MDEwOlJlcG9zaXRvcnkyODgxMDI4Mg==",
  "name": "158212",
  "full_name": "Mooophy/158212",
  "private": false,
  "owner": {
    "login": "Mooophy",
    "id": 5942966,
    "node_id": "MDQ6VXNlcjU5NDI5NjY=",
    "avatar_url": "https://avatars0.githubusercontent.com/u/5942966?v=4",
    "gravatar_id": "",
    "url": "https://api.github.com/users/Mooophy",
    "html_url": "https://github.com/Mooophy",
    "followers_url": "https://api.github.com/users/Mooophy/followers",
    "following_url": "https://api.github.com/users/Mooophy/following{/other_user}",
    "gists_url": "https://api.github.com/users/Mooophy/gists{/gist_id}",
    "starred_url": "https://api.github.com/users/Mooophy/starred{/owner}{/repo}",
    "subscriptions_url": "https://api.github.com/users/Mooophy/subscriptions",
    "organizations_url": "https://api.github.com/users/Mooophy/orgs",
    "repos_url": "https://api.github.com/users/Mooophy/repos",
    "events_url": "https://api.github.com/users/Mooophy/events{/privacy}",
    "received_events_url": "https://api.github.com/users/Mooophy/received_events",
    "type": "User",
    "site_admin": false
  },
  "html_url": "https://github.com/Mooophy/158212",
  "description": null,
  "fork": false,
  "url": "https://api.github.com/repos/Mooophy/158212",
  "forks_url": "https://api.github.com/repos/Mooophy/158212/forks",
  "keys_url": "https://api.github.com/repos/Mooophy/158212/keys{/key_id}",
  "collaborators_url": "https://api.github.com/repos/Mooophy/158212/collaborators{/collaborator}",
  "teams_url": "https://api.github.com/repos/Mooophy/158212/teams",
  "hooks_url": "https://api.github.com/repos/Mooophy/158212/hooks",
  "issue_events_url": "https://api.github.com/repos/Mooophy/158212/issues/events{/number}",
  "events_url": "https://api.github.com/repos/Mooophy/158212/events",
  "assignees_url": "https://api.github.com/repos/Mooophy/158212/assignees{/user}",
  "branches_url": "https://api.github.com/repos/Mooophy/158212/branches{/branch}",
  "tags_url": "https://api.github.com/repos/Mooophy/158212/tags",
  "blobs_url": "https://api.github.com/repos/Mooophy/158212/git/blobs{/sha}",
  "git_tags_url": "https://api.github.com/repos/Mooophy/158212/git/tags{/sha}",
  "git_refs_url": "https://api.github.com/repos/Mooophy/158212/git/refs{/sha}",
  "trees_url": "https://api.github.com/repos/Mooophy/158212/git/trees{/sha}",
  "statuses_url": "https://api.github.com/repos/Mooophy/158212/statuses/{sha}",
  "languages_url": "https://api.github.com/repos/Mooophy/158212/languages",
  "stargazers_url": "https://api.github.com/repos/Mooophy/158212/stargazers",
  "contributors_url": "https://api.github.com/repos/Mooophy/158212/contributors",
  "subscribers_url": "https://api.github.com/repos/Mooophy/158212/subscribers",
  "subscription_url": "https://api.github.com/repos/Mooophy/158212/subscription",
  "commits_url": "https://api.github.com/repos/Mooophy/158212/commits{/sha}",
  "git_commits_url": "https://api.github.com/repos/Mooophy/158212/git/commits{/sha}",
  "comments_url": "https://api.github.com/repos/Mooophy/158212/comments{/number}",
  "issue_comment_url": "https://api.github.com/repos/Mooophy/158212/issues/comments{/number}",
  "contents_url": "https://api.github.com/repos/Mooophy/158212/contents/{+path}",
  "compare_url": "https://api.github.com/repos/Mooophy/158212/compare/{base}...{head}",
  "merges_url": "https://api.github.com/repos/Mooophy/158212/merges",
  "archive_url": "https://api.github.com/repos/Mooophy/158212/{archive_format}{/ref}",
  "downloads_url": "https://api.github.com/repos/Mooophy/158212/downloads",
  "issues_url": "https://api.github.com/repos/Mooophy/158212/issues{/number}",
  "pulls_url": "https://api.github.com/repos/Mooophy/158212/pulls{/number}",
  "milestones_url": "https://api.github.com/repos/Mooophy/158212/milestones{/number}",
  "notifications_url": "https://api.github.com/repos/Mooophy/158212/notifications{?since,all,participating}",
  "labels_url": "https://api.github.com/repos/Mooophy/158212/labels{/name}",
  "releases_url": "https://api.github.com/repos/Mooophy/158212/releases{/id}",
  "deployments_url": "https://api.github.com/repos/Mooophy/158212/deployments",
  "created_at": "2015-01-05T11:42:01Z",
  "updated_at": "2015-07-25T21:45:01Z",
  "pushed_at": "2015-06-18T04:14:25Z",
  "git_url": "git://github.com/Mooophy/158212.git",
  "ssh_url": "git@github.com:Mooophy/158212.git",
  "clone_url": "https://github.com/Mooophy/158212.git",
  "svn_url": "https://github.com/Mooophy/158212",
  "homepage": "",
  "size": 35748,
  "stargazers_count": 1,
  "watchers_count": 1,
  "language": "C#",
  "has_issues": true,
  "has_projects": true,
  "has_downloads": true,
  "has_wiki": true,
  "has_pages": false,
  "forks_count": 0,
  "mirror_url": null,
  "archived": false,
  "open_issues_count": 2,
  "license": {
    "key": "mit",
    "name": "MIT License",
    "spdx_id": "MIT",
    "url": "https://api.github.com/licenses/mit",
    "node_id": "MDc6TGljZW5zZTEz"
  },
  "forks": 0,
  "open_issues": 2,
  "watchers": 1,
  "default_branch": "master"
}, {
  "id": 31251157,
  "node_id": "MDEwOlJlcG9zaXRvcnkzMTI1MTE1Nw==",
  "name": "158337",
  "full_name": "Mooophy/158337",
  "private": false,
  "owner": {
    "login": "Mooophy",
    "id": 5942966,
    "node_id": "MDQ6VXNlcjU5NDI5NjY=",
    "avatar_url": "https://avatars0.githubusercontent.com/u/5942966?v=4",
    "gravatar_id": "",
    "url": "https://api.github.com/users/Mooophy",
    "html_url": "https://github.com/Mooophy",
    "followers_url": "https://api.github.com/users/Mooophy/followers",
    "following_url": "https://api.github.com/users/Mooophy/following{/other_user}",
    "gists_url": "https://api.github.com/users/Mooophy/gists{/gist_id}",
    "starred_url": "https://api.github.com/users/Mooophy/starred{/owner}{/repo}",
    "subscriptions_url": "https://api.github.com/users/Mooophy/subscriptions",
    "organizations_url": "https://api.github.com/users/Mooophy/orgs",
    "repos_url": "https://api.github.com/users/Mooophy/repos",
    "events_url": "https://api.github.com/users/Mooophy/events{/privacy}",
    "received_events_url": "https://api.github.com/users/Mooophy/received_events",
    "type": "User",
    "site_admin": false
  },
  "html_url": "https://github.com/Mooophy/158337",
  "description": "database",
  "fork": false,
  "url": "https://api.github.com/repos/Mooophy/158337",
  "forks_url": "https://api.github.com/repos/Mooophy/158337/forks",
  "keys_url": "https://api.github.com/repos/Mooophy/158337/keys{/key_id}",
  "collaborators_url": "https://api.github.com/repos/Mooophy/158337/collaborators{/collaborator}",
  "teams_url": "https://api.github.com/repos/Mooophy/158337/teams",
  "hooks_url": "https://api.github.com/repos/Mooophy/158337/hooks",
  "issue_events_url": "https://api.github.com/repos/Mooophy/158337/issues/events{/number}",
  "events_url": "https://api.github.com/repos/Mooophy/158337/events",
  "assignees_url": "https://api.github.com/repos/Mooophy/158337/assignees{/user}",
  "branches_url": "https://api.github.com/repos/Mooophy/158337/branches{/branch}",
  "tags_url": "https://api.github.com/repos/Mooophy/158337/tags",
  "blobs_url": "https://api.github.com/repos/Mooophy/158337/git/blobs{/sha}",
  "git_tags_url": "https://api.github.com/repos/Mooophy/158337/git/tags{/sha}",
  "git_refs_url": "https://api.github.com/repos/Mooophy/158337/git/refs{/sha}",
  "trees_url": "https://api.github.com/repos/Mooophy/158337/git/trees{/sha}",
  "statuses_url": "https://api.github.com/repos/Mooophy/158337/statuses/{sha}",
  "languages_url": "https://api.github.com/repos/Mooophy/158337/languages",
  "stargazers_url": "https://api.github.com/repos/Mooophy/158337/stargazers",
  "contributors_url": "https://api.github.com/repos/Mooophy/158337/contributors",
  "subscribers_url": "https://api.github.com/repos/Mooophy/158337/subscribers",
  "subscription_url": "https://api.github.com/repos/Mooophy/158337/subscription",
  "commits_url": "https://api.github.com/repos/Mooophy/158337/commits{/sha}",
  "git_commits_url": "https://api.github.com/repos/Mooophy/158337/git/commits{/sha}",
  "comments_url": "https://api.github.com/repos/Mooophy/158337/comments{/number}",
  "issue_comment_url": "https://api.github.com/repos/Mooophy/158337/issues/comments{/number}",
  "contents_url": "https://api.github.com/repos/Mooophy/158337/contents/{+path}",
  "compare_url": "https://api.github.com/repos/Mooophy/158337/compare/{base}...{head}",
  "merges_url": "https://api.github.com/repos/Mooophy/158337/merges",
  "archive_url": "https://api.github.com/repos/Mooophy/158337/{archive_format}{/ref}",
  "downloads_url": "https://api.github.com/repos/Mooophy/158337/downloads",
  "issues_url": "https://api.github.com/repos/Mooophy/158337/issues{/number}",
  "pulls_url": "https://api.github.com/repos/Mooophy/158337/pulls{/number}",
  "milestones_url": "https://api.github.com/repos/Mooophy/158337/milestones{/number}",
  "notifications_url": "https://api.github.com/repos/Mooophy/158337/notifications{?since,all,participating}",
  "labels_url": "https://api.github.com/repos/Mooophy/158337/labels{/name}",
  "releases_url": "https://api.github.com/repos/Mooophy/158337/releases{/id}",
  "deployments_url": "https://api.github.com/repos/Mooophy/158337/deployments",
  "created_at": "2015-02-24T08:36:48Z",
  "updated_at": "2015-07-25T21:46:13Z",
  "pushed_at": "2015-04-06T02:19:21Z",
  "git_url": "git://github.com/Mooophy/158337.git",
  "ssh_url": "git@github.com:Mooophy/158337.git",
  "clone_url": "https://github.com/Mooophy/158337.git",
  "svn_url": "https://github.com/Mooophy/158337",
  "homepage": null,
  "size": 5436,
  "stargazers_count": 1,
  "watchers_count": 1,
  "language": null,
  "has_issues": true,
  "has_projects": true,
  "has_downloads": true,
  "has_wiki": true,
  "has_pages": false,
  "forks_count": 0,
  "mirror_url": null,
  "archived": false,
  "open_issues_count": 0,
  "license": {
    "key": "mit",
    "name": "MIT License",
    "spdx_id": "MIT",
    "url": "https://api.github.com/licenses/mit",
    "node_id": "MDc6TGljZW5zZTEz"
  },
  "forks": 0,
  "open_issues": 0,
  "watchers": 1,
  "default_branch": "master"
}, {
  "id": 27262898,
  "node_id": "MDEwOlJlcG9zaXRvcnkyNzI2Mjg5OA==",
  "name": "159201",
  "full_name": "Mooophy/159201",
  "private": false,
  "owner": {
    "login": "Mooophy",
    "id": 5942966,
    "node_id": "MDQ6VXNlcjU5NDI5NjY=",
    "avatar_url": "https://avatars0.githubusercontent.com/u/5942966?v=4",
    "gravatar_id": "",
    "url": "https://api.github.com/users/Mooophy",
    "html_url": "https://github.com/Mooophy",
    "followers_url": "https://api.github.com/users/Mooophy/followers",
    "following_url": "https://api.github.com/users/Mooophy/following{/other_user}",
    "gists_url": "https://api.github.com/users/Mooophy/gists{/gist_id}",
    "starred_url": "https://api.github.com/users/Mooophy/starred{/owner}{/repo}",
    "subscriptions_url": "https://api.github.com/users/Mooophy/subscriptions",
    "organizations_url": "https://api.github.com/users/Mooophy/orgs",
    "repos_url": "https://api.github.com/users/Mooophy/repos",
    "events_url": "https://api.github.com/users/Mooophy/events{/privacy}",
    "received_events_url": "https://api.github.com/users/Mooophy/received_events",
    "type": "User",
    "site_admin": false
  },
  "html_url": "https://github.com/Mooophy/159201",
  "description": null,
  "fork": false,
  "url": "https://api.github.com/repos/Mooophy/159201",
  "forks_url": "https://api.github.com/repos/Mooophy/159201/forks",
  "keys_url": "https://api.github.com/repos/Mooophy/159201/keys{/key_id}",
  "collaborators_url": "https://api.github.com/repos/Mooophy/159201/collaborators{/collaborator}",
  "teams_url": "https://api.github.com/repos/Mooophy/159201/teams",
  "hooks_url": "https://api.github.com/repos/Mooophy/159201/hooks",
  "issue_events_url": "https://api.github.com/repos/Mooophy/159201/issues/events{/number}",
  "events_url": "https://api.github.com/repos/Mooophy/159201/events",
  "assignees_url": "https://api.github.com/repos/Mooophy/159201/assignees{/user}",
  "branches_url": "https://api.github.com/repos/Mooophy/159201/branches{/branch}",
  "tags_url": "https://api.github.com/repos/Mooophy/159201/tags",
  "blobs_url": "https://api.github.com/repos/Mooophy/159201/git/blobs{/sha}",
  "git_tags_url": "https://api.github.com/repos/Mooophy/159201/git/tags{/sha}",
  "git_refs_url": "https://api.github.com/repos/Mooophy/159201/git/refs{/sha}",
  "trees_url": "https://api.github.com/repos/Mooophy/159201/git/trees{/sha}",
  "statuses_url": "https://api.github.com/repos/Mooophy/159201/statuses/{sha}",
  "languages_url": "https://api.github.com/repos/Mooophy/159201/languages",
  "stargazers_url": "https://api.github.com/repos/Mooophy/159201/stargazers",
  "contributors_url": "https://api.github.com/repos/Mooophy/159201/contributors",
  "subscribers_url": "https://api.github.com/repos/Mooophy/159201/subscribers",
  "subscription_url": "https://api.github.com/repos/Mooophy/159201/subscription",
  "commits_url": "https://api.github.com/repos/Mooophy/159201/commits{/sha}",
  "git_commits_url": "https://api.github.com/repos/Mooophy/159201/git/commits{/sha}",
  "comments_url": "https://api.github.com/repos/Mooophy/159201/comments{/number}",
  "issue_comment_url": "https://api.github.com/repos/Mooophy/159201/issues/comments{/number}",
  "contents_url": "https://api.github.com/repos/Mooophy/159201/contents/{+path}",
  "compare_url": "https://api.github.com/repos/Mooophy/159201/compare/{base}...{head}",
  "merges_url": "https://api.github.com/repos/Mooophy/159201/merges",
  "archive_url": "https://api.github.com/repos/Mooophy/159201/{archive_format}{/ref}",
  "downloads_url": "https://api.github.com/repos/Mooophy/159201/downloads",
  "issues_url": "https://api.github.com/repos/Mooophy/159201/issues{/number}",
  "pulls_url": "https://api.github.com/repos/Mooophy/159201/pulls{/number}",
  "milestones_url": "https://api.github.com/repos/Mooophy/159201/milestones{/number}",
  "notifications_url": "https://api.github.com/repos/Mooophy/159201/notifications{?since,all,participating}",
  "labels_url": "https://api.github.com/repos/Mooophy/159201/labels{/name}",
  "releases_url": "https://api.github.com/repos/Mooophy/159201/releases{/id}",
  "deployments_url": "https://api.github.com/repos/Mooophy/159201/deployments",
  "created_at": "2014-11-28T10:47:08Z",
  "updated_at": "2018-05-08T03:52:13Z",
  "pushed_at": "2015-11-01T21:51:45Z",
  "git_url": "git://github.com/Mooophy/159201.git",
  "ssh_url": "git@github.com:Mooophy/159201.git",
  "clone_url": "https://github.com/Mooophy/159201.git",
  "svn_url": "https://github.com/Mooophy/159201",
  "homepage": "",
  "size": 8900,
  "stargazers_count": 4,
  "watchers_count": 4,
  "language": "C++",
  "has_issues": true,
  "has_projects": true,
  "has_downloads": true,
  "has_wiki": true,
  "has_pages": false,
  "forks_count": 2,
  "mirror_url": null,
  "archived": false,
  "open_issues_count": 0,
  "license": {
    "key": "mit",
    "name": "MIT License",
    "spdx_id": "MIT",
    "url": "https://api.github.com/licenses/mit",
    "node_id": "MDc6TGljZW5zZTEz"
  },
  "forks": 2,
  "open_issues": 0,
  "watchers": 4,
  "default_branch": "master"
}, {
  "id": 37102470,
  "node_id": "MDEwOlJlcG9zaXRvcnkzNzEwMjQ3MA==",
  "name": "159233",
  "full_name": "Mooophy/159233",
  "private": false,
  "owner": {
    "login": "Mooophy",
    "id": 5942966,
    "node_id": "MDQ6VXNlcjU5NDI5NjY=",
    "avatar_url": "https://avatars0.githubusercontent.com/u/5942966?v=4",
    "gravatar_id": "",
    "url": "https://api.github.com/users/Mooophy",
    "html_url": "https://github.com/Mooophy",
    "followers_url": "https://api.github.com/users/Mooophy/followers",
    "following_url": "https://api.github.com/users/Mooophy/following{/other_user}",
    "gists_url": "https://api.github.com/users/Mooophy/gists{/gist_id}",
    "starred_url": "https://api.github.com/users/Mooophy/starred{/owner}{/repo}",
    "subscriptions_url": "https://api.github.com/users/Mooophy/subscriptions",
    "organizations_url": "https://api.github.com/users/Mooophy/orgs",
    "repos_url": "https://api.github.com/users/Mooophy/repos",
    "events_url": "https://api.github.com/users/Mooophy/events{/privacy}",
    "received_events_url": "https://api.github.com/users/Mooophy/received_events",
    "type": "User",
    "site_admin": false
  },
  "html_url": "https://github.com/Mooophy/159233",
  "description": null,
  "fork": false,
  "url": "https://api.github.com/repos/Mooophy/159233",
  "forks_url": "https://api.github.com/repos/Mooophy/159233/forks",
  "keys_url": "https://api.github.com/repos/Mooophy/159233/keys{/key_id}",
  "collaborators_url": "https://api.github.com/repos/Mooophy/159233/collaborators{/collaborator}",
  "teams_url": "https://api.github.com/repos/Mooophy/159233/teams",
  "hooks_url": "https://api.github.com/repos/Mooophy/159233/hooks",
  "issue_events_url": "https://api.github.com/repos/Mooophy/159233/issues/events{/number}",
  "events_url": "https://api.github.com/repos/Mooophy/159233/events",
  "assignees_url": "https://api.github.com/repos/Mooophy/159233/assignees{/user}",
  "branches_url": "https://api.github.com/repos/Mooophy/159233/branches{/branch}",
  "tags_url": "https://api.github.com/repos/Mooophy/159233/tags",
  "blobs_url": "https://api.github.com/repos/Mooophy/159233/git/blobs{/sha}",
  "git_tags_url": "https://api.github.com/repos/Mooophy/159233/git/tags{/sha}",
  "git_refs_url": "https://api.github.com/repos/Mooophy/159233/git/refs{/sha}",
  "trees_url": "https://api.github.com/repos/Mooophy/159233/git/trees{/sha}",
  "statuses_url": "https://api.github.com/repos/Mooophy/159233/statuses/{sha}",
  "languages_url": "https://api.github.com/repos/Mooophy/159233/languages",
  "stargazers_url": "https://api.github.com/repos/Mooophy/159233/stargazers",
  "contributors_url": "https://api.github.com/repos/Mooophy/159233/contributors",
  "subscribers_url": "https://api.github.com/repos/Mooophy/159233/subscribers",
  "subscription_url": "https://api.github.com/repos/Mooophy/159233/subscription",
  "commits_url": "https://api.github.com/repos/Mooophy/159233/commits{/sha}",
  "git_commits_url": "https://api.github.com/repos/Mooophy/159233/git/commits{/sha}",
  "comments_url": "https://api.github.com/repos/Mooophy/159233/comments{/number}",
  "issue_comment_url": "https://api.github.com/repos/Mooophy/159233/issues/comments{/number}",
  "contents_url": "https://api.github.com/repos/Mooophy/159233/contents/{+path}",
  "compare_url": "https://api.github.com/repos/Mooophy/159233/compare/{base}...{head}",
  "merges_url": "https://api.github.com/repos/Mooophy/159233/merges",
  "archive_url": "https://api.github.com/repos/Mooophy/159233/{archive_format}{/ref}",
  "downloads_url": "https://api.github.com/repos/Mooophy/159233/downloads",
  "issues_url": "https://api.github.com/repos/Mooophy/159233/issues{/number}",
  "pulls_url": "https://api.github.com/repos/Mooophy/159233/pulls{/number}",
  "milestones_url": "https://api.github.com/repos/Mooophy/159233/milestones{/number}",
  "notifications_url": "https://api.github.com/repos/Mooophy/159233/notifications{?since,all,participating}",
  "labels_url": "https://api.github.com/repos/Mooophy/159233/labels{/name}",
  "releases_url": "https://api.github.com/repos/Mooophy/159233/releases{/id}",
  "deployments_url": "https://api.github.com/repos/Mooophy/159233/deployments",
  "created_at": "2015-06-09T01:20:48Z",
  "updated_at": "2018-05-08T03:52:52Z",
  "pushed_at": "2015-06-09T03:07:02Z",
  "git_url": "git://github.com/Mooophy/159233.git",
  "ssh_url": "git@github.com:Mooophy/159233.git",
  "clone_url": "https://github.com/Mooophy/159233.git",
  "svn_url": "https://github.com/Mooophy/159233",
  "homepage": null,
  "size": 288,
  "stargazers_count": 2,
  "watchers_count": 2,
  "language": "Assembly",
  "has_issues": true,
  "has_projects": true,
  "has_downloads": true,
  "has_wiki": true,
  "has_pages": false,
  "forks_count": 0,
  "mirror_url": null,
  "archived": false,
  "open_issues_count": 0,
  "license": {
    "key": "mit",
    "name": "MIT License",
    "spdx_id": "MIT",
    "url": "https://api.github.com/licenses/mit",
    "node_id": "MDc6TGljZW5zZTEz"
  },
  "forks": 0,
  "open_issues": 0,
  "watchers": 2,
  "default_branch": "master"
},...
"""
