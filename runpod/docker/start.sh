#!/usr/bin/env bash
# Prepare operator access, then run the requested workload as PID 1.
#
# The base image is a plain CUDA image, so nothing else authorizes the operator
# key or starts a listener. Without this the Pod exits immediately and the stage
# command packets cannot reach it.
set -euo pipefail
umask 077

install -d -m 700 /root/.ssh
if [[ -n "${PUBLIC_KEY:-}" ]]; then
  printf '%s\n' "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
  chmod 600 /root/.ssh/authorized_keys
fi

# Remote sessions do not inherit image ENV. preflight_remote.py reads
# VOXDELTA_CODE_SHA256 and fails closed when it is empty, so publish the
# runtime and identity variables through pam_env instead.
{
  printf 'PATH=%s\n' "$PATH"
  printf 'PYTHONPATH=%s\n' "${PYTHONPATH:-}"
  printf 'LD_LIBRARY_PATH=%s\n' "${LD_LIBRARY_PATH:-}"
  printf 'VOXDELTA_CODE_SHA256=%s\n' "${VOXDELTA_CODE_SHA256:-}"
  printf 'VOXDELTA_GIT_COMMIT=%s\n' "${VOXDELTA_GIT_COMMIT:-}"
} > /etc/environment
chmod 644 /etc/environment

ssh-keygen -A
install -d -m 755 /run/sshd

exec "$@"
