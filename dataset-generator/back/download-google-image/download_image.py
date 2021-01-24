from simple_image_download import simple_image_download as simp
import argparse

def main(search_query, number_of_downloaded_images):
	simp().download(search_query, number_of_downloaded_images)

if __name__ == '__main__':
    # arguments parsing
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('-search', '--search_query', type=str, help='search string to use in searching in Google Search Image', default = 'nothing')
    argparser.add_argument('-n', '--number_of_downloaded_images', type=int, help='number of images to download', default=10)

    args = argparser.parse_args()

    main(args.search_query, args.number_of_downloaded_images)

