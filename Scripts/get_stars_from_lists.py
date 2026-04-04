import requests
from bs4 import BeautifulSoup
import pandas as pd

def get_star_lists(user):
    url = f'https://github.com/{user}?tab=stars'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    return [a['href'].split('/')[-1] for a in soup.select('a[href^="/stars/{0}/lists/"]'.format(user))]

def get_final_page_num(user, star_list_name):
    url = f'https://github.com/stars/{user}/lists/{star_list_name}?page=1'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    page_numbers = [int(a['href'].split('page=')[-1]) for a in soup.select('a[href^="/stars/{0}/lists/{1}?page="]'.format(user, star_list_name))]
    return max(page_numbers) if page_numbers else 1

def get_starred_repos(user, star_list_name, page):
    url = f'https://github.com/stars/{user}/lists/{star_list_name}?page={page}'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    repos = []
    repo_elements = soup.find_all("a", href=True)
    for repo_element in repo_elements:
        href = repo_element.get("href")
        if href.endswith('/stargazers'):
            user, repo = href.split('/')[1:3]
            repos.append((user, repo))
    return repos

def main():
    github_username = 'evdcush'
    print('gettings star lists')
    star_lists = get_star_lists(github_username)
    print('got star lists')
    df_columns = ['user', 'repo'] + star_lists
    df = pd.DataFrame(columns=df_columns)

    for star_list in star_lists:
        final_page_num = get_final_page_num(github_username, star_list)
        print(f'{final_page_num = } ')

        for page_num in range(1, final_page_num + 1):
            print(f'{page_num = }')
            starred_repos = get_starred_repos(github_username, star_list, page_num)

            for user, repo in starred_repos:
                if not ((df['user'] == user) & (df['repo'] == repo)).any():
                    print(f"star: {user}/{repo}")
                    new_row = {col: 0 for col in star_lists}
                    new_row['user'] = user
                    new_row['repo'] = repo
                    new_row[star_list] = 1
                    #df = df.append(new_row, ignore_index=True)
                    df = pd.concat([df, pd.DataFrame.from_records([new_row])], ignore_index=True)
                else:
                    df.loc[(df['user'] == user) & (df['repo'] == repo), star_list] = 1

    try:
        df.to_csv('my_stars.csv', index=False)
        print('all good')
        from IPython import embed; embed();
    except:
        print('problem')
        from IPython import embed; embed();

if __name__ == '__main__':
    main()
