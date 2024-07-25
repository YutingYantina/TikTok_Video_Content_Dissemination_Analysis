from mitmproxy import http
import json
import csv

def response(flow: http.HTTPFlow) -> None:
    if "https://www.douyin.com/aweme/v1/web/aweme/post/" in flow.request.url:
        try:
            response_data = json.loads(flow.response.content)
            aweme_list = response_data.get("aweme_list", [])
            with open('statistics.csv', mode='a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(['admire_count', 'collect_count', 'comment_count', 'digg_count', 'play_count', 'share_count'])
                for aweme in aweme_list:
                    statistics = aweme.get("statistics", {})
                    admire_count = statistics.get("admire_count", 0)
                    collect_count = statistics.get("collect_count", 0)
                    comment_count = statistics.get("comment_count", 0)
                    digg_count = statistics.get("digg_count", 0)
                    play_count = statistics.get("play_count", 0)
                    share_count = statistics.get("share_count", 0)
                    writer.writerow([admire_count, collect_count, comment_count, digg_count, play_count, share_count])
            print("Statistic saved")
        except ValueError:
            print("No response")
