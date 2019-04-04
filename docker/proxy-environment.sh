. /etc/profile

if test -z "$https_proxy" ; then
  echo "http_proxy and https_proxy should be set via docker build-args or by ENV" >&2
  exit 1
fi

cacerts=/usr/local/share/ca-certificates/Custom.crt
if ! test -f "$cacerts" ; then
  echo "Custom certificates is not set. Did you run 'proxy-certificates'?" >&2
  exit 1
fi

if test -d "$JAVA_HOME" ; then

  keystore=$JAVA_HOME/lib/security/cacerts
  pems_dir=/tmp/pems
  rm -rf "$pems_dir" 2>/dev/null || true
  mkdir "$pems_dir"
  (
  cd "$pems_dir"
  awk 'BEGIN {c=0;doPrint=0;} /END CERT/ {print > "cert." c ".pem";doPrint=0;} /BEGIN CERT/{c++;doPrint=1;} { if(doPrint == 1) {print > "cert." c ".pem"} }' < $cacerts
  for f in `ls cert.*.pem`; do
    keytool -import -trustcacerts -noprompt -keystore "$keystore" -alias "`basename $f`" -file "$f" -storepass changeit;
  done
  )
  rm -rf "$pems_dir"
else
  echo "JAVA_HOME is not set" >&2
fi

PROXY_HOST=`echo $https_proxy | sed 's@.*//\(.*\):.*@\1@'`
PROXY_PORT=`echo $https_proxy | sed 's@.*//.*:\(.*\)@\1@'`
{
echo "export http_proxy=$http_proxy"
echo "export https_proxy=$https_proxy"
echo "export HTTP_PROXY=$http_proxy"
echo "export HTTPS_PROXY=$https_proxy"
echo "export GRADLE_OPTS='-Dorg.gradle.daemon=false -Dandroid.builder.sdkDownload=true -Dorg.gradle.jvmargs=-Xmx2048M -Dhttp.proxyHost=$PROXY_HOST -Dhttp.proxyPort=$PROXY_PORT -Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT'"
echo "export MAVEN_OPTS='-Dhttps.proxyHost=$PROXY_HOST -Dhttps.proxyPort=$PROXY_PORT -Dmaven.wagon.http.ssl.insecure=true'"
echo "export no_proxy=localhost,127.0.0.0,127.0.1.1,127.0.1.1,.huawei.com"
} >>/etc/profile

if test -f /etc/sudoers ; then
{
echo "Defaults        env_keep += \"http_proxy HTTP_PROXY https_proxy HTTPS_PROXY no_proxy GRADLE_OPTS MAVEN_OPTS\""
} >>/etc/sudoers
else
  echo "/etc/sudoers is not updated" >&2
fi

if test -f /etc/wgetrc; then
  echo ca_certificate=/etc/ssl/certs/ca-certificates.crt >> /etc/wgetrc
else
  echo "/etc/wgetrc is not updated" >&2
fi

mkdir /root/.android/
cat >/root/.android/androidtool.cfg <<EOF
http.proxyHost=$PROXY_HOST
http.proxyPort=$PROXY_PORT
https.proxyHost=$PROXY_HOST
https.proxyPort=$PROXY_PORT
EOF

# TODO: Handle Pythonish import ssl; ssl._create_default_https_context = ssl._create_unverified_context;
# TODO: Handle Pythonish certifi internal certificates

