import requests
import zipfile
import click

@click.command()
@click.argument('url')
@click.argument('filename', type=click.Path())

def download_file(url, filename):
    """Download data"""
    print('Downloading from {} to {}'.format(url,filename))
    responce = requests.get(url)
    with open(filename, 'wb') as ofile:
        ofile.write(responce.content)

if __name__ == '__main__':
    download_file()

