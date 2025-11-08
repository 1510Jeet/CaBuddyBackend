import requests
from bs4 import BeautifulSoup
import json
import time
from urllib.parse import urljoin

def extract_main_content(soup, url):
    """Extract only main content from Tally Help page"""
    
    # Try multiple possible content containers in order of preference
    entry_content = None
    
    # Try standard content divs first
    for selector in [
        ('div', {'class': 'entry-content'}),
        ('div', {'id': 'articleDetailDiv'}),
        ('div', {'class': 'articleDetailDiv'}),
        ('div', {'class': 'kb-ajax-load-container'}),
        ('div', {'class': 'wpb-content-wrapper'}),  # New: for WPBakery pages
        ('div', {'class': 'vc_row'}),  # New: Visual Composer rows
        ('main', {}),
        ('article', {}),
    ]:
        entry_content = soup.find(selector[0], selector[1])
        if entry_content:
            break
    
    if not entry_content:
        return None
    
    # Get title
    title_tag = soup.find('title')
    title = title_tag.get_text().strip() if title_tag else ''
    
    # Get breadcrumbs for context
    breadcrumbs = soup.find('div', class_='header-breadcrumbs')
    breadcrumb_text = breadcrumbs.get_text(separator=' > ', strip=True) if breadcrumbs else ''
    
    # Check for iframes and fetch their content
    iframe_content = []
    base_domain = "https://help.tallysolutions.com/"
    for iframe in entry_content.find_all('iframe'):
        iframe_src = iframe.get('src', '')
        if iframe_src:
            try:
                # Make absolute URL if needed
                iframe_src = urljoin(base_domain, iframe_src)
                
                # Only fetch iframes from help.tallysolutions.com
                if iframe_src.startswith(base_domain):
                    print(f"    → Fetching iframe: {iframe_src}")
                    iframe_response = requests.get(iframe_src, timeout=15)
                    if iframe_response.status_code == 200:
                        iframe_soup = BeautifulSoup(iframe_response.content, 'html.parser')
                        # Try to find content in iframe
                        iframe_body = (
                            iframe_soup.find('div', id='articleDetailDiv') or
                            iframe_soup.find('body')
                        )
                        if iframe_body:
                            iframe_text = iframe_body.get_text(separator='\n', strip=True)
                            iframe_content.append(iframe_text)
            except Exception as e:
                print(f"    ✗ Iframe fetch error: {e}")
    
    # Remove unwanted elements from content
    for unwanted in entry_content(['script', 'style', 'nav', 'footer', 'aside', 'header']):
        unwanted.decompose()
    
    # Also remove specific unwanted classes/ids
    for unwanted_class in entry_content.find_all(class_=['sidebar', 'menu', 'navigation', 'social', 'single-post-sidebar']):
        unwanted_class.decompose()
    
    # Extract text
    text = entry_content.get_text(separator='\n', strip=True)
    
    # Add iframe content
    if iframe_content:
        text = text + '\n\n' + '\n\n'.join(iframe_content)
    
    # Clean up text
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    cleaned_text = '\n'.join(lines)
    
    # Skip if content is too short (likely just navigation)
    if len(cleaned_text) < 100:
        return None
    
    # Extract images with context
    images = []
    for img in entry_content.find_all('img'):
        img_data = {
            'src': img.get('src', ''),
            'alt': img.get('alt', ''),
        }
        if img_data['src']:
            images.append(img_data)
    
    return {
        'url': url,
        'title': title,
        'breadcrumbs': breadcrumb_text,
        'content': cleaned_text,
        'images': images
    }


def recursive_scrape(start_url, max_pages=10000, output_file='tally_scraped_content.json'):
    """Recursively scrape and save incrementally"""
    
    base_url = "https://help.tallysolutions.com/"
    
    # Pre-populate with URLs to always skip
    visited = {
        "https://help.tallysolutions.com/advanced-search/",
        "https://help.tallysolutions.com/tallyhelp-videos/",
        "https://help.tallysolutions.com/developer-reference/"
    }
    
    # File extensions to skip
    skip_ext = ['.jpg', '.jpeg', '.png', '.gif', '.pdf', '.zip', 
                '.mp4', '.avi', '.css', '.js', '.svg', '.webp']
    
    to_visit = [start_url]
    to_visit_set = {start_url}  # Track what's queued
    pages_scraped = 0
    
    # Open file in write mode at start
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('[\n')  # Start JSON array
    
    print(f"Starting crawl from: {start_url}")
    print(f"Max pages: {max_pages}")
    print(f"Saving to: {output_file}\n")
    
    while to_visit and pages_scraped < max_pages:
        url = to_visit.pop(0)
        to_visit_set.discard(url)  # Remove from tracking set
        
        # Skip if already visited or has skip extension
        if url in visited or any(url.lower().endswith(e) for e in skip_ext):
            visited.add(url)
            continue
        
        print(f"[{pages_scraped+1}] Crawling: {url}")
        visited.add(url)
        
        try:
            response = requests.get(url, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Extract content
                page_data = extract_main_content(soup, url)
                
                if page_data and page_data['content']:
                    # Append to file immediately
                    with open(output_file, 'a', encoding='utf-8') as f:
                        if pages_scraped > 0:
                            f.write(',\n')  # Add comma between objects
                        json.dump(page_data, f, ensure_ascii=False, indent=2)
                    
                    pages_scraped += 1
                    print(f"  ✓ Saved! {len(page_data['content'])} chars, {len(page_data['images'])} images")
                else:
                    print(f"  ✗ No content found")
                
                # Find more links to crawl
                for link in soup.find_all('a', href=True):
                    href = link['href']
                    
                    # Build full URL safely using current base
                    full_url = urljoin(base_url, href)
                    
                    # Only allow URLs that START WITH base_url
                    if full_url.startswith(base_url):
                        clean_url = full_url.split('?')[0].split('#')[0].rstrip('/')
                        
                        # Check if NOT visited AND NOT already queued AND NOT a file
                        if (clean_url not in visited and 
                            clean_url not in to_visit_set and
                            not any(clean_url.lower().endswith(ext) for ext in skip_ext)):
                            to_visit.append(clean_url)
                            to_visit_set.add(clean_url)
            
            time.sleep(0.3)
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
        
        # Progress update
        if pages_scraped % 50 == 0 and pages_scraped > 0:
            print(f"\n📊 Progress: {pages_scraped} pages scraped, {len(to_visit)} queued\n")
    
    # Close JSON array
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write('\n]')
    
    print(f"\n Scraping complete: {pages_scraped} pages saved to {output_file}")
    return pages_scraped

# Run
pages_count = recursive_scrape("https://help.tallysolutions.com/tally-prime/", max_pages=10000)
