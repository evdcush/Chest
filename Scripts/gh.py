#!/usr/bin/env python
"""
gh
##
Do stuff with git and github.

Use PyGithub:
https://pygithub.readthedocs.io/en/latest/introduction.html
"""
import os
import sys
import argparse

from IPython import embed

from github import Github
import yaml


# Get access token
# ================
tokens_path = os.getenv("HOME") + '/.ghtokenz.yaml'
assert os.path.exists(tokens_path)

# Load the yaml file.
with open(tokens_path, 'r') as token_file:
    tokens = yaml.safe_load(token_file)

# Get the desired token.
#token_enc = tokens['kagami']
#token = bytes.fromhex(token_enc).decode()
token = tokens['kagami']


# Instantiate GH
# ==============
gh = Github(token)

# --------------------------------------------------------------------------- #

# Sandbox
#repos = g.get_user().get_repos()  # type: PaginatedList ....


embed()



'''
# Play with your GH objects:
for repo in g.get_user().get_repos():
    print(repo.name)
    #repo.edit(has_wiki=False)
    # to see all the available attributes and methods

print(dir(repo))
'''


'''
TODOS
#####

awesomes
========
Something to download, organize, and update awesomes.

mirrorer
========
Something to mirror repos instead of fork.


'''


'''
#=============================================================================#
#                                                                             #
#                     ██████  ██    ██ ███    ███ ██████                      #
#                     ██   ██ ██    ██ ████  ████ ██   ██                     #
#                     ██   ██ ██    ██ ██ ████ ██ ██████                      #
#                     ██   ██ ██    ██ ██  ██  ██ ██                          #
#                     ██████   ██████  ██      ██ ██                          #
#                                                                             #
#=============================================================================#

'''
In [2]: dir(gh)
Out[2]:
['FIX_REPO_GET_GIT_REF',
 '_Github__requester',
 '__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getstate__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 'create_from_raw_data',
 'dump',
 'get_app',
 'get_emojis',
 'get_events',
 'get_gist',
 'get_gists',
 'get_gitignore_template',
 'get_gitignore_templates',
 'get_hook',
 'get_hooks',
 'get_license',
 'get_licenses',
 'get_oauth_application',
 'get_organization',
 'get_organizations',
 'get_project',
 'get_project_column',
 'get_rate_limit',
 'get_repo',
 'get_repos',
 'get_user',
 'get_user_by_id',
 'get_users',
 'load',
 'oauth_scopes',
 'per_page',
 'rate_limiting',
 'rate_limiting_resettime',
 'render_markdown',
 'search_code',
 'search_commits',
 'search_issues',
 'search_repositories',
 'search_topics',
 'search_users']
'''


n [1]: len(repos)
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[1], line 1
----> 1 len(repos)

TypeError: object of type 'PaginatedList' has no len()

In [2]: repos
Out[2]: <github.PaginatedList.PaginatedList at 0x7f3984903b50>

In [3]: dir(repos)
Out[3]:
['_PaginatedListBase__elements',
 '_PaginatedListBase__fetchToIndex',
 '_PaginatedList__contentClass',
 '_PaginatedList__firstParams',
 '_PaginatedList__firstUrl',
 '_PaginatedList__headers',
 '_PaginatedList__list_item',
 '_PaginatedList__nextParams',
 '_PaginatedList__nextUrl',
 '_PaginatedList__parseLinkHeader',
 '_PaginatedList__requester',
 '_PaginatedList__reverse',
 '_PaginatedList__totalCount',
 '_Slice',
 '__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__getitem__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__iter__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 '_couldGrow',
 '_fetchNextPage',
 '_getLastPageUrl',
 '_grow',
 '_isBiggerThan',
 '_reversed',
 'get_page',
 'reversed',
 'totalCount']

In [4]: repos_list = list(repos)

In [5]: repos.totalCount
Out[5]: 351

In [6]: repo = repos.get_page()
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
Cell In[6], line 1
----> 1 repo = repos.get_page()

TypeError: PaginatedList.get_page() missing 1 required positional argument: 'page'

In [7]: repo = repos.get_page(0)

In [8]: repo
Out[8]:
[Repository(full_name="evdcush/3detr"),
 Repository(full_name="evdcush/ablog"),
 Repository(full_name="evdcush/ad-rss-lib"),
 Repository(full_name="evdcush/Agenda"),
 Repository(full_name="evdcush/agentcraft"),
 Repository(full_name="evdcush/ai-collection"),
 Repository(full_name="evdcush/AI_Resources"),
 Repository(full_name="evdcush/alien"),
 Repository(full_name="evdcush/Almagest"),
 Repository(full_name="evdcush/alpha-zero-general"),
 Repository(full_name="evdcush/alphafold"),
 Repository(full_name="evdcush/alphaicon"),
 Repository(full_name="evdcush/alphastar"),
 Repository(full_name="evdcush/alphatensor"),
 Repository(full_name="evdcush/alternative-front-ends"),
 Repository(full_name="evdcush/alto"),
 Repository(full_name="evdcush/AnimatedDrawings"),
 Repository(full_name="evdcush/animegan2-pytorch"),
 Repository(full_name="evdcush/AnimeGANv2"),
 Repository(full_name="evdcush/Anki-decks"),
 Repository(full_name="evdcush/ANML"),
 Repository(full_name="evdcush/applied-ml"),
 Repository(full_name="evdcush/ArchiveBox"),
 Repository(full_name="evdcush/ARF-svox2"),
 Repository(full_name="evdcush/arnheim"),
 Repository(full_name="evdcush/ArxParse"),
 Repository(full_name="evdcush/Ascent-Coding-Challenge-2018"),
 Repository(full_name="evdcush/audiolm-pytorch"),
 Repository(full_name="evdcush/autocast"),
 Repository(full_name="evdcush/autoDocstring")]

In [9]: rep = repos_list[0]

In [10]: rep
Out[10]: Repository(full_name="evdcush/3detr")

In [11]: dir(rep)
Out[11]:
['CHECK_AFTER_INIT_FLAG',
 '_CompletableGithubObject__complete',
 '_CompletableGithubObject__completed',
 '_GithubObject__makeSimpleAttribute',
 '_GithubObject__makeSimpleListAttribute',
 '_GithubObject__makeTransformedAttribute',
 '_Repository__create_pull',
 '_Repository__create_pull_1',
 '_Repository__create_pull_2',
 '__class__',
 '__delattr__',
 '__dict__',
 '__dir__',
 '__doc__',
 '__eq__',
 '__format__',
 '__ge__',
 '__getattribute__',
 '__gt__',
 '__hash__',
 '__init__',
 '__init_subclass__',
 '__le__',
 '__lt__',
 '__module__',
 '__ne__',
 '__new__',
 '__reduce__',
 '__reduce_ex__',
 '__repr__',
 '__setattr__',
 '__sizeof__',
 '__str__',
 '__subclasshook__',
 '__weakref__',
 '_allow_forking',
 '_allow_merge_commit',
 '_allow_rebase_merge',
 '_allow_squash_merge',
 '_archive_url',
 '_archived',
 '_assignees_url',
 '_blobs_url',
 '_branches_url',
 '_clone_url',
 '_collaborators_url',
 '_comments_url',
 '_commits_url',
 '_compare_url',
 '_completeIfNeeded',
 '_completeIfNotSet',
 '_contents_url',
 '_contributors_url',
 '_created_at',
 '_default_branch',
 '_delete_branch_on_merge',
 '_deployments_url',
 '_description',
 '_downloads_url',
 '_events_url',
 '_fork',
 '_forks',
 '_forks_count',
 '_forks_url',
 '_full_name',
 '_git_commits_url',
 '_git_refs_url',
 '_git_tags_url',
 '_git_url',
 '_has_downloads',
 '_has_issues',
 '_has_pages',
 '_has_projects',
 '_has_wiki',
 '_headers',
 '_homepage',
 '_hooks_url',
 '_html_url',
 '_hub',
 '_id',
 '_identity',
 '_initAttributes',
 '_is_template',
 '_issue_comment_url',
 '_issue_events_url',
 '_issues_url',
 '_keys_url',
 '_labels_url',
 '_language',
 '_languages_url',
 '_legacy_convert_issue',
 '_makeBoolAttribute',
 '_makeClassAttribute',
 '_makeDatetimeAttribute',
 '_makeDictAttribute',
 '_makeDictOfStringsToClassesAttribute',
 '_makeFloatAttribute',
 '_makeIntAttribute',
 '_makeListOfClassesAttribute',
 '_makeListOfDictsAttribute',
 '_makeListOfIntsAttribute',
 '_makeListOfListOfStringsAttribute',
 '_makeListOfStringsAttribute',
 '_makeStringAttribute',
 '_makeTimestampAttribute',
 '_master_branch',
 '_merges_url',
 '_milestones_url',
 '_mirror_url',
 '_name',
 '_network_count',
 '_notifications_url',
 '_open_issues',
 '_open_issues_count',
 '_organization',
 '_owner',
 '_parent',
 '_parentUrl',
 '_permissions',
 '_private',
 '_pulls_url',
 '_pushed_at',
 '_rawData',
 '_releases_url',
 '_requester',
 '_size',
 '_source',
 '_ssh_url',
 '_stargazers_count',
 '_stargazers_url',
 '_statuses_url',
 '_storeAndUseAttributes',
 '_subscribers_count',
 '_subscribers_url',
 '_subscription_url',
 '_svn_url',
 '_tags_url',
 '_teams_url',
 '_topics',
 '_trees_url',
 '_updated_at',
 '_url',
 '_useAttributes',
 '_visibility',
 '_watchers',
 '_watchers_count',
 'add_to_collaborators',
 'allow_forking',
 'allow_merge_commit',
 'allow_rebase_merge',
 'allow_squash_merge',
 'archive_url',
 'archived',
 'assignees_url',
 'blobs_url',
 'branches_url',
 'clone_url',
 'collaborators_url',
 'comments_url',
 'commits_url',
 'compare',
 'compare_url',
 'contents_url',
 'contributors_url',
 'create_autolink',
 'create_check_run',
 'create_check_suite',
 'create_deployment',
 'create_file',
 'create_fork',
 'create_git_blob',
 'create_git_commit',
 'create_git_ref',
 'create_git_release',
 'create_git_tag',
 'create_git_tag_and_release',
 'create_git_tree',
 'create_hook',
 'create_issue',
 'create_key',
 'create_label',
 'create_milestone',
 'create_project',
 'create_pull',
 'create_repository_dispatch',
 'create_secret',
 'create_source_import',
 'created_at',
 'default_branch',
 'delete',
 'delete_branch_on_merge',
 'delete_file',
 'delete_secret',
 'deployments_url',
 'description',
 'disable_automated_security_fixes',
 'disable_vulnerability_alert',
 'downloads_url',
 'edit',
 'enable_automated_security_fixes',
 'enable_vulnerability_alert',
 'etag',
 'events_url',
 'fork',
 'forks',
 'forks_count',
 'forks_url',
 'full_name',
 'get__repr__',
 'get_archive_link',
 'get_artifact',
 'get_artifacts',
 'get_assignees',
 'get_autolinks',
 'get_branch',
 'get_branches',
 'get_check_run',
 'get_check_suite',
 'get_clones_traffic',
 'get_codescan_alerts',
 'get_collaborator_permission',
 'get_collaborators',
 'get_comment',
 'get_comments',
 'get_commit',
 'get_commits',
 'get_contents',
 'get_contributors',
 'get_deployment',
 'get_deployments',
 'get_dir_contents',
 'get_download',
 'get_downloads',
 'get_events',
 'get_forks',
 'get_git_blob',
 'get_git_commit',
 'get_git_matching_refs',
 'get_git_ref',
 'get_git_refs',
 'get_git_tag',
 'get_git_tree',
 'get_hook',
 'get_hooks',
 'get_issue',
 'get_issues',
 'get_issues_comments',
 'get_issues_event',
 'get_issues_events',
 'get_key',
 'get_keys',
 'get_label',
 'get_labels',
 'get_languages',
 'get_latest_release',
 'get_license',
 'get_milestone',
 'get_milestones',
 'get_network_events',
 'get_notifications',
 'get_pending_invitations',
 'get_projects',
 'get_public_key',
 'get_pull',
 'get_pulls',
 'get_pulls_comments',
 'get_pulls_review_comments',
 'get_readme',
 'get_release',
 'get_release_asset',
 'get_releases',
 'get_self_hosted_runner',
 'get_self_hosted_runners',
 'get_source_import',
 'get_stargazers',
 'get_stargazers_with_dates',
 'get_stats_code_frequency',
 'get_stats_commit_activity',
 'get_stats_contributors',
 'get_stats_participation',
 'get_stats_punch_card',
 'get_subscribers',
 'get_tags',
 'get_teams',
 'get_top_paths',
 'get_top_referrers',
 'get_topics',
 'get_views_traffic',
 'get_vulnerability_alert',
 'get_watchers',
 'get_workflow',
 'get_workflow_run',
 'get_workflow_runs',
 'get_workflows',
 'git_commits_url',
 'git_refs_url',
 'git_tags_url',
 'git_url',
 'has_downloads',
 'has_in_assignees',
 'has_in_collaborators',
 'has_issues',
 'has_pages',
 'has_projects',
 'has_wiki',
 'homepage',
 'hooks_url',
 'html_url',
 'id',
 'is_template',
 'issue_comment_url',
 'issue_events_url',
 'issues_url',
 'keys_url',
 'labels_url',
 'language',
 'languages_url',
 'last_modified',
 'legacy_search_issues',
 'mark_notifications_as_read',
 'master_branch',
 'merge',
 'merges_url',
 'milestones_url',
 'mirror_url',
 'name',
 'network_count',
 'notifications_url',
 'open_issues',
 'open_issues_count',
 'organization',
 'owner',
 'parent',
 'permissions',
 'private',
 'pulls_url',
 'pushed_at',
 'raw_data',
 'raw_headers',
 'releases_url',
 'remove_autolink',
 'remove_from_collaborators',
 'remove_invitation',
 'remove_self_hosted_runner',
 'rename_branch',
 'replace_topics',
 'setCheckAfterInitFlag',
 'size',
 'source',
 'ssh_url',
 'stargazers_count',
 'stargazers_url',
 'statuses_url',
 'subscribe_to_hub',
 'subscribers_count',
 'subscribers_url',
 'subscription_url',
 'svn_url',
 'tags_url',
 'teams_url',
 'topics',
 'trees_url',
 'unsubscribe_from_hub',
 'update',
 'update_check_suites_preferences',
 'update_file',
 'updated_at',
 'url',
 'visibility',
 'watchers',
 'watchers_count']

'''
