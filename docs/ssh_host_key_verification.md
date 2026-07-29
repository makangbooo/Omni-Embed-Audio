# Bitahub SSH host-key verification

This procedure is read-only. It does not connect from the Codex client, edit
`known_hosts`, accept a key, or disclose an SSH private key.

## Why the port is not the identity

Bitahub may assign a new external port when an instance changes. OpenSSH stores
non-default-port entries as `[hostname]:port`, so a port change alters the
lookup entry. It does not prove that a newly presented host key is legitimate.
The ED25519 fingerprint must be checked through the Bitahub Web console or
another independent trusted channel.

## Check from the Bitahub Web console

Open a terminal for the exact target instance through Bitahub's authenticated
Web console. From the repository root, run:

```bash
bash scripts/print_ssh_host_fingerprints.sh \
  SHA256:+lMykBVk8nCgA/Aav7/pG5AcS6UrJqy5OuKKLt+q8ZA
```

A local instance match must include both of these lines:

```text
HOST_KEY_FILE=/etc/ssh/ssh_host_ed25519_key.pub
VERIFY_RESULT=ED25519_MATCH
```

Also inspect `HOST_KEY_IDENTITY`: its second field must be exactly
`SHA256:+lMykBVk8nCgA/Aav7/pG5AcS6UrJqy5OuKKLt+q8ZA` and its algorithm must be
`ED25519`.

If the result is `ED25519_MISMATCH`, do not accept the client-observed key. If
no local host public key is visible, Bitahub may terminate SSH at an external
gateway. In that case, obtain the gateway fingerprint from the authenticated
Bitahub console or Bitahub support and compare the full SHA256 value. Do not
use `StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`, or an
unverified `ssh-keyscan` result as proof of identity.

## Current verified instance observation

On 2026-07-29 the authenticated Bitahub Web console reported:

- instance: `bitahub-a20633967503994880812290`;
- external endpoint: `xj-member.bitahub.com:42156`;
- local key: `/etc/ssh/ssh_host_ed25519_key.pub`;
- local ED25519 fingerprint:
  `SHA256:+lMykBVk8nCgA/Aav7/pG5AcS6UrJqy5OuKKLt+q8ZA`;
- repository: clean `repro/oea-full` at `4a7852e` before fast-forward.

The earlier client-observed `SHA256:etC2qN4P9phlmmEtoHi7hO3qfnmevz/u7bJv8Q1JwlM`
does not match this instance and must not be accepted for port `42156`.
The local key is verified out of band, but the external endpoint remains
unbound until its exact public key is installed for the new host/port tuple.

The following Web-console command prints a `known_hosts` candidate without
modifying any file:

```bash
awk 'NR == 1 {print "[xj-member.bitahub.com]:42156", $1, $2}' \
  /etc/ssh/ssh_host_ed25519_key.pub
```

Before installation, independently confirm that the candidate's fingerprint
is the verified `+lMy...` value. Adding it for port `42156` must not remove or
silently replace entries for other ports.

## After a verified match

Report the following without sending any password or private key:

- the exact `VERIFY_RESULT` line;
- the complete ED25519 `HOST_KEY_IDENTITY` line;
- the current external hostname and port shown by Bitahub.

Only after the match is independently confirmed should the stale entry for the
old exact `[hostname]:port` be replaced. Authentication is a separate check:
use a public key installed through Bitahub or another platform-approved login
method, and never send an SSH private key through chat.
