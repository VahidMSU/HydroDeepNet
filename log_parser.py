import re

log_file_path = '/data/SWATGenXApp/codes/web_application/logs/ciwre-bae_access.log'
output_file_path = '/data/SWATGenXApp/codes/web_application/logs/parsed_log.csv'

# Regular expression to parse log entries
log_pattern = re.compile(
    r'(?P<ip>\d+\.\d+\.\d+\.\d+) - - \[(?P<date>.*?)\] "(?P<method>\w+) (?P<url>.*?) HTTP/1.1" (?P<status>\d+) (?P<size>\d+) "(?P<referrer>.*?)" "(?P<user_agent>.*?)"'
)

with (open(log_file_path, 'r') as log_file, open(output_file_path, 'w') as output_file):
    output_file.write('IP,Date,Method,URL,Status,Size,Referrer,User-Agent\n')
    for line in log_file:
        if match := log_pattern.match(line):
            log_data = match.groupdict()
            output_file.write(
                f"{log_data['ip']},{log_data['date']},{log_data['method']},{log_data['url']},{log_data['status']},{log_data['size']},{log_data['referrer']},{log_data['user_agent']}\n"
            )
