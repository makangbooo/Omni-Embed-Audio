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
  SHA256:etC2qN4P9phlmmEtoHi7hO3qfnmevz/u7bJv8Q1JwlM
```

A local instance match must include both of these lines:

```text
HOST_KEY_FILE=/etc/ssh/ssh_host_ed25519_key.pub
VERIFY_RESULT=ED25519_MATCH
```

Also inspect `HOST_KEY_IDENTITY`: its second field must be exactly
`SHA256:etC2qN4P9phlmmEtoHi7hO3qfnmevz/u7bJv8Q1JwlM` and its algorithm must be
`ED25519`.

If the result is `ED25519_MISMATCH`, do not accept the client-observed key. If
no local host public key is visible, Bitahub may terminate SSH at an external
gateway. In that case, obtain the gateway fingerprint from the authenticated
Bitahub console or Bitahub support and compare the full SHA256 value. Do not
use `StrictHostKeyChecking=no`, `UserKnownHostsFile=/dev/null`, or an
unverified `ssh-keyscan` result as proof of identity.

## After a verified match

Report the following without sending any password or private key:

- the exact `VERIFY_RESULT` line;
- the complete ED25519 `HOST_KEY_IDENTITY` line;
- the current external hostname and port shown by Bitahub.

Only after the match is independently confirmed should the stale entry for the
old exact `[hostname]:port` be replaced. Authentication is a separate check:
use a public key installed through Bitahub or another platform-approved login
method, and never send an SSH private key through chat.
