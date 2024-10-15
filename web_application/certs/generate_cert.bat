@echo off
REM Set the OPENSSL_CONF environment variable to the path of your openssl.cnf file
set OPENSSL_CONF=D:\MyDataBase\codes\web_application\certs\openssl.cnf

REM Generate private key
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\Library\bin\openssl.exe" genrsa -out private.key 2048

REM Generate certificate signing request (CSR)
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\Library\bin\openssl.exe" req -new -key private.key -out cert.csr -config "%OPENSSL_CONF%"

REM Generate self-signed certificate
"C:\Program Files\ArcGIS\Pro\bin\Python\envs\arcgispro-py3\Library\bin\openssl.exe" x509 -req -days 365 -in cert.csr -signkey private.key -out certificate.crt

pause
