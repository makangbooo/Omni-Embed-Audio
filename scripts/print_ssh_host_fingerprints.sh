#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 <expected-SHA256-fingerprint>" >&2
  echo "Example: $0 SHA256:base64-fingerprint" >&2
  exit 2
fi

EXPECTED_FINGERPRINT="$1"
if [[ ! "${EXPECTED_FINGERPRINT}" =~ ^SHA256:[A-Za-z0-9+/]+={0,2}$ ]]; then
  echo "[ERROR] Expected fingerprint must use OpenSSH SHA256 format." >&2
  exit 2
fi
if ! command -v ssh-keygen >/dev/null 2>&1; then
  echo "[ERROR] ssh-keygen is unavailable on this server." >&2
  exit 3
fi

shopt -s nullglob
PUBLIC_KEYS=(/etc/ssh/ssh_host_*_key.pub)
if [[ "${#PUBLIC_KEYS[@]}" -eq 0 ]]; then
  echo "[ERROR] No /etc/ssh/ssh_host_*_key.pub files are visible." >&2
  echo "[INFO] The platform may terminate SSH outside this instance." >&2
  echo "[INFO] Verify the fingerprint in the Bitahub console or with Bitahub support." >&2
  exit 4
fi

echo "VERIFICATION_SCOPE=read-only local SSH host public keys"
echo "HOSTNAME=$(hostname)"
echo "CHECKED_AT=$(date -Is)"
echo "EXPECTED_ED25519_FINGERPRINT=${EXPECTED_FINGERPRINT}"
echo "EXTERNAL_PORT_NOTE=Port forwarding may change; the host-key fingerprint is verified independently of the external port."

MATCHED=0
READABLE=0
for public_key in "${PUBLIC_KEYS[@]}"; do
  if [[ ! -r "${public_key}" ]]; then
    echo "HOST_KEY_UNREADABLE=${public_key}"
    continue
  fi
  READABLE="$((READABLE + 1))"
  identity="$(ssh-keygen -E sha256 -lf "${public_key}")"
  fingerprint="$(awk '{print $2}' <<<"${identity}")"
  echo "HOST_KEY_FILE=${public_key}"
  echo "HOST_KEY_IDENTITY=${identity}"
  if [[ "${public_key}" == "/etc/ssh/ssh_host_ed25519_key.pub" && \
        "${fingerprint}" == "${EXPECTED_FINGERPRINT}" ]]; then
    MATCHED=1
  fi
done

if [[ "${READABLE}" -eq 0 ]]; then
  echo "VERIFY_RESULT=NO_READABLE_HOST_PUBLIC_KEYS"
  exit 5
fi
if [[ "${MATCHED}" -ne 1 ]]; then
  echo "VERIFY_RESULT=ED25519_MISMATCH"
  echo "[ERROR] Do not accept or install the client-observed host key." >&2
  echo "[INFO] If Bitahub uses an SSH gateway, obtain the gateway fingerprint from Bitahub through an independent trusted channel." >&2
  exit 6
fi

echo "VERIFY_RESULT=ED25519_MATCH"
echo "[OK] The client-observed fingerprint matches this instance's ED25519 host public key."
