from __future__ import annotations

import re
from typing import Iterable, Optional
from urllib.parse import urlparse

# Role taxonomy used by the deployable gateway feature extractor.  Web3 drainer
# roles are retained, but the taxonomy is intentionally broader: interactive
# phishing kits often use identity providers, CAPTCHA / bot challenges, hosted
# forms, messaging webhooks, cloud APIs, object storage, payment processors, and
# low-cost hosting platforms.  The role labels are only coarse gateway-visible
# abstractions from SNI/passive DNS; URL paths, DOM, form fields, screenshots,
# and wallet/browser extension internals are not required.
WC_RE = re.compile(r"(^|\.)(walletconnect\.(com|org|network)|reown\.com|web3modal\.(com|org))$", re.I)
RPC_RE = re.compile(
    r"(^|\.)(infura\.io|alchemy\.com|alchemyapi\.io|quiknode\.pro|quicknode\.com|ankr\.com|publicnode\.com|llamarpc\.com|blastapi\.io|chainstack\.com|moralis\.io|drpc\.org|blockpi\.network)$|infura-router\.public\.blockchain-networks",
    re.I,
)
WALLET_VENDOR_RE = re.compile(
    r"(^|\.)(metamask\.io|metamask\.github\.io|api\.cx\.metamask\.io|cx\.metamask\.io|rabby\.io|phantom\.app|okx\.com|okex\.org|okxwallet\.com|okx-httpdns\.com|okx\.cab|okx\.ac|coinall\.ltd|wallet\.ouxyi\.cash|trustwallet\.com|rainbow\.me|coinbase\.com|walletlink\.org|safe\.global|gnosis-safe\.io|ledger\.com|ledgerwallet\.com|zerion\.io|onekey\.so|token\.im|imtoken\.com|unisat\.io|bitgetwallet\.io|bitkeep\.com|backpack\.app|exodus\.com)$",
    re.I,
)

IDENTITY_RE = re.compile(
    r"(^|\.)(accounts\.google\.com|login\.microsoftonline\.com|login\.live\.com|account\.live\.com|appleid\.apple\.com|facebook\.com|okta\.com|okta-emea\.com|auth0\.com|duosecurity\.com|onelogin\.com|pingidentity\.com|pingone\.com|sso\.[^.]+\.[^.]+|login\.[^.]+\.[^.]+|auth\.[^.]+\.[^.]+)$",
    re.I,
)
CAPTCHA_RE = re.compile(
    r"(^|\.)(recaptcha\.net|hcaptcha\.com|challenges\.cloudflare\.com|turnstile\.cloudflare\.com|arkoselabs\.com|funcaptcha\.com|datadome\.co|datadome\.com|perimeterx\.net|px-cloud\.net|kasada\.io)$",
    re.I,
)
PAYMENT_RE = re.compile(
    r"(^|\.)(stripe\.com|stripe\.network|paypal\.com|paypalobjects\.com|adyen\.com|braintreegateway\.com|braintree-api\.com|squareup\.com|squarecdn\.com|checkout\.com|klarna\.com|afterpay\.com|worldpay\.com|payoneer\.com|wise\.com|razorpay\.com|paystack\.com)$",
    re.I,
)
MESSAGING_API_RE = re.compile(
    r"(^|\.)(api\.telegram\.org|telegram\.org|discord\.com|discordapp\.com|slack\.com|hooks\.slack\.com|webhook\.site|pipedream\.net|ifttt\.com|maker\.ifttt\.com|ntfy\.sh|pushover\.net)$",
    re.I,
)
FORM_BACKEND_RE = re.compile(
    r"(^|\.)(formspree\.io|formsubmit\.co|getform\.io|formcarry\.com|staticforms\.xyz|web3forms\.com|emailjs\.com|jotform\.com|typeform\.com|tally\.so|forms\.gle|googleusercontent\.com|airtable\.com)$",
    re.I,
)
CLOUD_API_RE = re.compile(
    r"(^|\.)(firebaseio\.com|firebasedatabase\.app|firebaseapp\.com|firebase\.googleapis\.com|firestore\.googleapis\.com|supabase\.co|supabase\.in|appwrite\.io|appwrite\.global|airtable\.com|api\.notion\.com|notion\.com|zapier\.com|hooks\.zapier\.com|make\.com|integromat\.com|script\.google\.com|script\.googleusercontent\.com)$",
    re.I,
)
OBJECT_STORAGE_RE = re.compile(
    r"(^|\.)(storage\.googleapis\.com|s3\.amazonaws\.com|s3[.-][a-z0-9-]+\.amazonaws\.com|amazonaws\.com|blob\.core\.windows\.net|r2\.cloudflarestorage\.com|digitaloceanspaces\.com|backblazeb2\.com|wasabisys\.com)$",
    re.I,
)
EMAIL_DELIVERY_RE = re.compile(
    r"(^|\.)(sendgrid\.net|sendgrid\.com|mailgun\.org|mailgun\.net|mandrillapp\.com|mailchimp\.com|smtp2go\.com|brevo\.com|sendinblue\.com|postmarkapp\.com|sparkpostmail\.com)$",
    re.I,
)
FILE_STORAGE_RE = re.compile(
    r"(^|\.)(drive\.google\.com|docs\.google\.com|dropbox\.com|dropboxusercontent\.com|onedrive\.live\.com|1drv\.ms|box\.com|mega\.nz|mediafire\.com|wetransfer\.com)$",
    re.I,
)
URL_SHORTENER_RE = re.compile(
    r"(^|\.)(bit\.ly|tinyurl\.com|t\.co|cutt\.ly|rebrand\.ly|lnkd\.in|is\.gd|ow\.ly|shorturl\.at|rb\.gy|s\.id|buff\.ly)$",
    re.I,
)
HOSTING_PLATFORM_RE = re.compile(
    r"(^|\.)(vercel\.app|netlify\.app|pages\.dev|workers\.dev|github\.io|gitlab\.io|firebaseapp\.com|web\.app|herokuapp\.com|glitch\.me|replit\.app|render\.com|fly\.dev|surge\.sh|ngrok-free\.app|ngrok\.io|trycloudflare\.com|duckdns\.org|hopto\.org|no-ip\.org|serveo\.net|localtunnel\.me)$",
    re.I,
)
STATIC_RE = re.compile(
    r"(^|\.)(cdnjs\.cloudflare\.com|cloudflare\.com|jsdelivr\.net|unpkg\.com|code\.jquery\.com|jquery\.com|googleapis\.com|gstatic\.com|fontawesome\.com|bootstrapcdn\.com|cloudfront\.net|akamaihd\.net|fastly\.net|assets\.adobedtm\.com|fonts\.googleapis\.com|fonts\.gstatic\.com)$",
    re.I,
)
ANALYTICS_RE = re.compile(
    r"(^|\.)(google-analytics\.com|googletagmanager\.com|sentry\.io|datadoghq\.com|segment\.io|amplitude\.com|mixpanel\.com|intercom\.io|hotjar\.com|clarity\.ms|newrelic\.com|nr-data\.net|fullstory\.com|doubleclick\.net|googlesyndication\.com|googletagservices\.com|facebook\.net)$",
    re.I,
)
SOFTWARE_UPDATE_RE = re.compile(
    r"(^|\.)(clients[0-9]*\.google\.com|update\.googleapis\.com|safebrowsing\.googleapis\.com|edge\.microsoft\.com|msedge\.api\.cdp\.microsoft\.com|aus[0-9]*\.mozilla\.org|versioncheck-bg\.addons\.mozilla\.org|addons\.mozilla\.org|extensions\.gstatic\.com)$",
    re.I,
)

ROLE_PRIORITY = [
    "walletconnect",
    "rpc_provider",
    "wallet_vendor",
    "identity_provider",
    "captcha_challenge",
    "payment_provider",
    "messaging_api",
    "form_backend",
    "cloud_api",
    "object_storage",
    "email_delivery",
    "file_storage",
    "url_shortener",
    "hosting_platform",
    "first_party_site",
    "software_update",
    "third_party_static",
    "analytics_ads",
    "third_party_backend_or_other",
    "unknown",
]

NUISANCE_ROLES = {
    "wallet_vendor",
    "analytics_ads",
    "third_party_static",
    "software_update",
}

INTERACTION_ROLES = {
    "unknown",
    "first_party_site",
    "third_party_backend_or_other",
    "identity_provider",
    "captcha_challenge",
    "payment_provider",
    "messaging_api",
    "form_backend",
    "cloud_api",
    "object_storage",
    "email_delivery",
    "file_storage",
    "url_shortener",
    "hosting_platform",
    "rpc_provider",
    "walletconnect",
}


def normalize_host(host: Optional[str]) -> str:
    if not host:
        return ""
    host = host.strip().lower().strip(".")
    if ":" in host and not host.startswith("["):
        host = host.split(":", 1)[0]
    return host


def host_from_url(url: str) -> str:
    try:
        return normalize_host(urlparse(url).hostname or "")
    except Exception:
        return ""


def naive_registrable_domain(host: str) -> str:
    """Dependency-free approximation for role grouping.

    Production deployments should replace this with publicsuffix2/tldextract.
    Managed hosting names are intentionally kept at full-host granularity so all
    ``*.vercel.app`` or ``*.pages.dev`` sites do not collapse into one party.
    """
    host = normalize_host(host)
    parts = host.split(".")
    if len(parts) <= 2:
        return host
    managed_suffixes = (
        ".vercel.app",
        ".pages.dev",
        ".netlify.app",
        ".workers.dev",
        ".github.io",
        ".web.app",
        ".firebaseapp.com",
        ".glitch.me",
        ".replit.app",
        ".render.com",
        ".fly.dev",
    )
    if host.endswith(managed_suffixes):
        return host
    return ".".join(parts[-2:])


def is_first_party(host: str, primary_host: str) -> bool:
    host = normalize_host(host)
    primary_host = normalize_host(primary_host)
    if not host or not primary_host:
        return False
    if host == primary_host or host.endswith("." + primary_host):
        return True
    managed_suffixes = (
        ".vercel.app",
        ".pages.dev",
        ".netlify.app",
        ".workers.dev",
        ".github.io",
        ".web.app",
        ".firebaseapp.com",
        ".glitch.me",
        ".replit.app",
        ".render.com",
        ".fly.dev",
    )
    if primary_host.endswith(managed_suffixes):
        return False
    return naive_registrable_domain(host) == naive_registrable_domain(primary_host)


def classify_host(host: Optional[str], primary_host: Optional[str] = None, feature_mode: str = "clean") -> str:
    host = normalize_host(host)
    primary_host = normalize_host(primary_host)
    if not host:
        return "unknown"
    if WC_RE.search(host):
        return "walletconnect"
    if RPC_RE.search(host):
        return "rpc_provider"
    if WALLET_VENDOR_RE.search(host):
        return "wallet_vendor"
    if IDENTITY_RE.search(host):
        return "identity_provider"
    if CAPTCHA_RE.search(host):
        return "captcha_challenge"
    if PAYMENT_RE.search(host):
        return "payment_provider"
    if MESSAGING_API_RE.search(host):
        return "messaging_api"
    if FORM_BACKEND_RE.search(host):
        return "form_backend"
    if CLOUD_API_RE.search(host):
        return "cloud_api"
    if OBJECT_STORAGE_RE.search(host):
        return "object_storage"
    if EMAIL_DELIVERY_RE.search(host):
        return "email_delivery"
    if FILE_STORAGE_RE.search(host):
        return "file_storage"
    if URL_SHORTENER_RE.search(host):
        return "url_shortener"
    if primary_host and is_first_party(host, primary_host):
        return "first_party_site"
    if HOSTING_PLATFORM_RE.search(host):
        return "hosting_platform"
    if SOFTWARE_UPDATE_RE.search(host):
        return "software_update"
    if STATIC_RE.search(host):
        return "third_party_static"
    if ANALYTICS_RE.search(host):
        return "analytics_ads"
    return "third_party_backend_or_other"


def merge_roles(roles: Iterable[str]) -> str:
    roles = {r for r in roles if r}
    if not roles:
        return "unknown"
    for r in ROLE_PRIORITY:
        if r in roles:
            return r
    return "unknown"
