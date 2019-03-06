"""The MIT License (MIT)
Copyright (c) 2015 Ali Rasim Kocal <arkocal@gmail.com> arkocal.org
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE."""


import os
import re

import urllib2

class PageManager(object):
    """Class for managing pages."""

    def __init__(self):
        self.pages = {}

    def load_with_title(self, title):
        """Load the page with given title."""

        if title in self.pages.keys():
            return
        url = "http://en.wikipedia.org/wiki/" + title
        self.pages[title] = Page(url)

    def sort_page_relevance(self, title):
        """Return a sorted list of titles and relevance sorted by relevance
        Relevance is determined by the ratio of common links from both pages
        to total number of links in the second."""
        self.load_with_title(title)
        relevance = []
        page = self.pages[title]
        for t in self.pages:
            ratio = page.get_common_link_ratio(self.pages[t])
            relevance.append((t, ratio))
        relevance.sort(key= lambda x: x[1])
        relevance.reverse()
        return relevance



class Page(object):

    def __init__(self, url):
        self.url = url
        # http://en.wikipedia.org/wiki/TITLE#INNER_LINK
        self.title = url.split("/")[-1].split("#")[0]
        self.read_page()
        self.get_main_div()
        self.find_links()

    def read_page(self):
        """Read the page
        Tries to read from local disk first. If the file is not found,
        it is downloaded."""
        try:
            tempfile = open(self.title)
        except FileNotFoundError:
            #os.system("wget {}".format(self.url))
            tempfile = urllib2.open(self.url)
        self.lines = tempfile
        #tempfile.close()

    def get_main_div(self):
        self.main_div_content = []
        started = False

        print("fetching main div...")
        print(self.lines)

        for line in self.lines:
            if not started:
                if line.strip().startswith('<div id="bodyContent'):
                    self.main_div_content.append(line)
                    started = True
            else:
                if line.strip().startswith('<div id="mw-navigation'):
                    break
                self.main_div_content.append(line)

        print('Arrived here! 1')

        print(self.main_div_content)
        # Rip off first generic lines (links to search etc.)
        self.main_div_content = self.main_div_content[9:]

    def find_links(self):
        self.links = []
        for line in self.main_div_content:
            match = re.search('/wiki/([a-zA-Z_:]*)', line)
            if not match: continue
            if ":" not in match.group(1): self.links.append(match.group(1))

        print('Arrived here!')
        print(self.links)
        self.links = set(self.links)

    def find_common_links(self, page):
        common_links =[]
        for link in self.links:
            if link in page.links:
                common_links.append(link)
        return common_links

    def count_common_links(self, page):
        return len(self.find_common_links(page))

    def get_common_link_ratio(self, page):
        count = self.count_common_links(page)
        return count/len(page.links)

if __name__ == "__main__":
    page_manager = PageManager()
    example_titles = ["Yalova", "Istanbul", "Berlin", "Turkey", "Germany",
                      "Cat", "Dog", "Wolf", "Banana", "Watermelon",
                      "Mathematics", "Physics", "Chemistry", "Biology"]

    for title in example_titles:
        page_manager.load_with_title(title)

    print("<table style='border-collapse: collapse;border: 1px solid black'>")
    for title in example_titles:
        print("<tr>")
        print("<td style='background-color:gray; padding:8px;"+
               "border: 1px solid black'>" + title + "</td>")
        relevance = page_manager.sort_page_relevance(title)
        for t, r in relevance[1:]:
            p1 = page_manager.pages[title]
            p2 = page_manager.pages[t]
            m = "&#10".join(p1.find_common_links(p2))
            cell = ("<td style='padding:8px; border:1px dashed black' title={matches}"+\
                   ">{title} {r:.2f}</td>").format(title=t[:3], r=r*100, matches=m)
            print(cell)
        print("</tr>")
    print("</table>")
    print("####")
    print(page_manager.pages["Yalova"].links)
    print()
    print(page_manager.pages["Yalova"].find_common_links(page_manager.pages["Istanbul"]))
