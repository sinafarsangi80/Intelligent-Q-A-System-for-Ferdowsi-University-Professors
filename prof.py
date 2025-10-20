import scrapy
from urllib.parse import urljoin

class UmProfessorsSpider(scrapy.Spider):
    name = 'um_professors'
    allowed_domains = ['um.ac.ir', 'prof.um.ac.ir']
    start_urls = ['https://um.ac.ir/members/professors/list.html']

    def parse(self, response):
        rows = response.css('table.table tbody tr')
        for row in rows:
            professor = {
                'نام خانوادگی': row.css('td:nth-child(2)::text').get(),
                'نام': row.css('td:nth-child(3)::text').get(),
                'دانشکده': row.css('td:nth-child(4)::text').get(),
                'گروه آموزشی': row.css('td:nth-child(5)::text').get(),
                'وضعیت اشتغال': row.css('td:nth-child(6)::text').get(),
                'پست الکترونیکی': row.css('td:nth-child(7)::text').get(),
                'صفحه شخصی': row.css('td:nth-child(8) a::attr(href)').get(),
            }

            personal_page = professor['صفحه شخصی']
            if personal_page:
                yield response.follow(personal_page, callback=self.parse_personal_page, meta={'professor': professor})
            else:
                yield professor

        # Pagination handling
        next_page = response.css('ul.pagination li.next a::attr(href)').get()
        if next_page:
            yield response.follow(next_page, callback=self.parse)

    def parse_personal_page(self, response):
        professor = response.meta['professor']
        
        # Determine page language
        lang = response.xpath('//html/@lang').get()

        # Publications link text based on language
        if lang == 'fa':
            pub_label = 'انتشارات'
        else:
            pub_label = 'Publications'

        # Find the correct publications link
        pub_link = response.xpath(f'//div[@class="link text" and text()="{pub_label}"]/parent::a/@href').get()

        if pub_link:
            pub_link = response.urljoin(pub_link)
            yield scrapy.Request(pub_link, callback=self.parse_publications, meta={'professor': professor})
        else:
            professor['publications'] = []
            yield professor

            
    def parse_publications(self, response):
        professor = response.meta['professor']
        publications = []

        for item in response.css('div.pitems div.item'):
            title = item.css('h4.pubtitle::text').get()
            authors = item.css('div.pubauthor strong a::text').getall()
            journal = item.css('div.pubcite p::text').get()
            article_link = item.css('div.pubassets a::attr(href)').get()

            publication = {
                'title': title,
                'authors': authors,
                'journal': journal,
                'article_link': article_link
            }
            publications.append(publication)

        professor['publications'] = publications
        yield professor

