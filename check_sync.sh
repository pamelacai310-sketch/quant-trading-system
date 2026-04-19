#!/bin/bash
# GitHub 同步状态检查脚本

echo "🔍 检查与GitHub的同步状态..."
echo ""

# 检查git状态
git_status=$(git status --porcelain)
if [ -z "$git_status" ]; then
    echo "✅ 本地工作区干净"
else
    echo "⚠️  有未提交的更改:"
    echo "$git_status"
fi

echo ""
echo "📊 与远程仓库的对比:"
git fetch origin
local_commit=$(git rev-parse main)
remote_commit=$(git rev-parse origin/main)

if [ "$local_commit" = "$remote_commit" ]; then
    echo "✅ 本地与远程完全同步"
    echo "   提交: ${local_commit:0:8}"
else
    echo "⚠️  本地与远程不同步"
    echo "   本地: ${local_commit:0:8}"
    echo "   远程: ${remote_commit:0:8}"
    echo ""
    echo "💡 运行以下命令同步:"
    echo "   git pull origin main    # 拉取远程更改"
    echo "   git push origin main    # 推送本地更改"
fi

echo ""
echo "📦 仓库信息:"
gh repo view $(git remote get-url origin | sed 's/.*github.com[:/]\(.*\)\.git/\1/') --json url,visibility,defaultBranchRef --jq '"URL: " + .url + "\n可见性: " + .visibility + "\n默认分支: " + .defaultBranchRef.name'
