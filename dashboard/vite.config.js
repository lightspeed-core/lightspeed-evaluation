import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { execSync, spawn } from 'child_process'
import fs from 'fs'
import path from 'path'
import YAML from 'js-yaml'

function resolveEnvPath(envVar, fallback) {
  const raw = process.env[envVar] || ''
  if (!raw) return fallback ? path.resolve(__dirname, fallback) : ''
  return path.isAbsolute(raw) ? raw : path.resolve(__dirname, raw)
}

const evalOutputDir = resolveEnvPath('LS_EVAL_REPORTS_PATH', './results')
const evalDataFile = resolveEnvPath('LS_EVAL_DATA_PATH', './eval.yaml')

function evalDataPlugin() {
  const activeRuns = new Map()
  let runIdCounter = 0

  return {
    name: 'eval-data',
    configureServer(server) {
      server.middlewares.use('/results', (req, res, next) => {
        const filePath = path.join(evalOutputDir, decodeURIComponent(req.url))
        if (fs.existsSync(filePath) && fs.statSync(filePath).isFile()) {
          const ext = path.extname(filePath)
          const types = { '.csv': 'text/csv', '.yaml': 'text/yaml', '.yml': 'text/yaml', '.png': 'image/png', '.json': 'application/json' }
          res.setHeader('Content-Type', types[ext] || 'application/octet-stream')
          fs.createReadStream(filePath).pipe(res)
        } else {
          next()
        }
      })

      server.middlewares.use('/api/manifest', (_req, res) => {
        const dirExists = fs.existsSync(evalOutputDir)
        let files = []
        if (dirExists) {
          files = fs.readdirSync(evalOutputDir)
            .filter(f => /^evaluation_\d{8}_\d{6}_detailed\.csv$/.test(f))
            .sort()
        }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ files, reportsDir: evalOutputDir, reportsDirExists: dirExists }))
      })

      server.middlewares.use('/api/amended-files', (_req, res) => {
        let files = []
        try {
          files = fs.readdirSync(evalOutputDir)
            .filter(f => /_amended_\d{8}_\d{6}\.yaml$/.test(f))
            .sort()
        } catch { /* dir may not exist */ }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(files))
      })

      server.middlewares.use('/api/eval-graphs', (_req, res) => {
        const graphsDir = path.join(evalOutputDir, 'graphs')
        const map = {}
        try {
          for (const f of fs.readdirSync(graphsDir)) {
            const m = f.match(/^evaluation_(\d{8}_\d{6})_.+\.png$/)
            if (m) {
              if (!map[m[1]]) map[m[1]] = []
              map[m[1]].push(f)
            }
          }
        } catch { /* ignore if dir doesn't exist */ }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(map))
      })

      server.middlewares.use('/api/eval-summaries', (_req, res) => {
        const map = {}
        try {
          for (const f of fs.readdirSync(evalOutputDir)) {
            const m = f.match(/^evaluation_(\d{8}_\d{6})_summary\.json$/)
            if (m) {
              try {
                const content = JSON.parse(fs.readFileSync(path.join(evalOutputDir, f), 'utf-8'))
                const model = content?.configuration?.api?.model
                if (model) map[m[1]] = { model }
              } catch { /* skip malformed JSON */ }
            }
          }
        } catch { /* ignore if dir doesn't exist */ }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(map))
      })

      server.middlewares.use('/api/run-config', (_req, res) => {
        const systemConfig = process.env.LS_EVAL_SYSTEM_CFG_PATH || ''
        const apiKey = process.env.API_KEY || ''
        let tags = []
        try {
          const content = fs.readFileSync(evalDataFile, 'utf-8')
          const parsed = YAML.load(content)
          if (Array.isArray(parsed)) {
            const tagSet = new Set()
            for (const group of parsed) {
              tagSet.add(group.tag || 'eval')
            }
            tags = [...tagSet].sort()
          }
        } catch { /* ignore */ }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ systemConfig, apiKey: !!apiKey, evalDataPath: evalDataFile, tags }))
      })

      server.middlewares.use('/api/run-eval', (req, res, next) => {
        if (req.method !== 'POST') return next()
        let body = ''
        req.on('data', c => { body += c })
        req.on('end', () => {
          let parsed
          try {
            parsed = JSON.parse(body)
          } catch {
            res.statusCode = 400
            res.end(JSON.stringify({ error: 'Invalid JSON' }))
            return
          }
          const { systemConfig, tag } = parsed
          if (!systemConfig) {
            res.statusCode = 400
            res.end(JSON.stringify({ error: 'Missing systemConfig' }))
            return
          }

          const id = String(++runIdCounter)
          const cwd = __dirname
          const outputPath = evalOutputDir

          const args = [
            '--system-config', systemConfig,
            '--eval-data', evalDataFile,
            '--output-dir', outputPath,
          ]
          if (tag && tag !== '__all__') {
            args.push('--tags', tag)
          }

          const child = spawn('lightspeed-eval', args, {
            cwd,
            env: { ...process.env, PYTHONUNBUFFERED: '1' },
          })

          const run = {
            id, pid: child.pid, tag: tag || 'all', systemConfig,
            startTime: Date.now(), output: '', exitCode: null,
            status: 'running', listeners: new Set(), child,
          }
          activeRuns.set(id, run)

          child.stdout.on('data', (d) => {
            const text = d.toString()
            run.output += text
            for (const fn of run.listeners) fn('output', text)
          })
          child.stderr.on('data', (d) => {
            const text = d.toString()
            run.output += text
            for (const fn of run.listeners) fn('output', text)
          })
          child.on('close', (code) => {
            run.exitCode = code ?? 1
            run.status = 'done'
            for (const fn of run.listeners) fn('exit', run.exitCode)
            run.listeners.clear()
          })
          child.on('error', (err) => {
            const text = `Error: ${err.message}\n`
            run.output += text
            for (const fn of run.listeners) fn('output', text)
            run.exitCode = 1
            run.status = 'done'
            for (const fn of run.listeners) fn('exit', 1)
            run.listeners.clear()
          })

          res.setHeader('Content-Type', 'application/json')
          res.end(JSON.stringify({ id, pid: child.pid }))
        })
      })

      server.middlewares.use('/api/running-evals', (_req, res) => {
        const runs = []
        for (const [, run] of activeRuns) {
          runs.push({
            id: run.id, pid: run.pid, tag: run.tag,
            startTime: run.startTime, status: run.status,
            exitCode: run.exitCode, source: 'web',
          })
        }
        try {
          const out = execSync('pgrep -af "[l]ightspeed-eval" 2>/dev/null || true').toString().trim()
          if (out) {
            const knownPids = new Set([...activeRuns.values()].map(r => r.pid))
            for (const line of out.split('\n')) {
              const match = line.match(/^(\d+)\s+(.+)/)
              if (!match) continue
              const pid = parseInt(match[1])
              const cmd = match[2]
              if (knownPids.has(pid)) continue
              // Filter out non-lightspeed-eval processes
              if (/pgrep|\/bin\/sh/.test(cmd)) continue
              // Only match if lightspeed-eval is the actual command, not just part of a path
              if (!/(?:^|\s|\/)(lightspeed-eval)(?:\s|$)/.test(cmd)) continue
              runs.push({
                id: `ext-${pid}`, pid, tag: cmd,
                startTime: null, status: 'running',
                exitCode: null, source: 'external',
              })
            }
          }
        } catch { /* ignore */ }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify(runs))
      })

      server.middlewares.use('/api/eval-stream', (req, res) => {
        const id = decodeURIComponent(req.url.slice(1))
        const run = activeRuns.get(id)
        if (!run) {
          res.statusCode = 404
          res.end(JSON.stringify({ error: 'Run not found' }))
          return
        }
        res.setHeader('Content-Type', 'text/event-stream')
        res.setHeader('Cache-Control', 'no-cache')
        res.setHeader('Connection', 'keep-alive')
        res.setHeader('X-Accel-Buffering', 'no')
        res.flushHeaders()

        const send = (event, data) => {
          res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`)
        }

        if (run.output) {
          send('output', { text: run.output })
        }

        if (run.status === 'done') {
          send('exit', { code: run.exitCode })
          res.end()
          return
        }

        const listener = (type, data) => {
          if (type === 'output') send('output', { text: data })
          else if (type === 'exit') { send('exit', { code: data }); res.end() }
        }
        run.listeners.add(listener)

        res.on('close', () => {
          run.listeners.delete(listener)
        })
      })

      server.middlewares.use('/api/stop-eval', (req, res, next) => {
        if (req.method !== 'POST') return next()
        const id = decodeURIComponent(req.url.slice(1))
        const run = activeRuns.get(id)
        if (!run || run.status !== 'running') {
          res.statusCode = 404
          res.end(JSON.stringify({ error: 'Run not found or already done' }))
          return
        }
        run.child.kill()
        run.output += '\n--- Stopped by user ---\n'
        for (const fn of run.listeners) fn('output', '\n--- Stopped by user ---\n')
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ ok: true }))
      })

      server.middlewares.use('/api/system-config', (req, res) => {
        const cfgPath = process.env.LS_EVAL_SYSTEM_CFG_PATH || ''
        const resolved = cfgPath
          ? path.isAbsolute(cfgPath)
            ? cfgPath
            : path.resolve(__dirname, cfgPath)
          : ''

        if (req.method === 'POST') {
          let body = ''
          req.on('data', c => { body += c })
          req.on('end', () => {
            try {
              const { content } = JSON.parse(body)
              if (!resolved) {
                res.statusCode = 400
                res.setHeader('Content-Type', 'application/json')
                res.end(JSON.stringify({ error: 'LS_EVAL_SYSTEM_CFG_PATH not set' }))
                return
              }
              fs.writeFileSync(resolved, content, 'utf-8')
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ ok: true, content }))
            } catch (err) {
              res.statusCode = 500
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ error: err.message }))
            }
          })
          return
        }

        let content = ''
        if (resolved) {
          try {
            content = fs.readFileSync(resolved, 'utf-8')
          } catch { /* ignore */ }
        }
        const runEnabled = ['true', '1'].includes((process.env.LS_EVAL_DASHBOARD_RUN_ENABLED || '').toLowerCase())
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ set: !!cfgPath, path: cfgPath, content, runEnabled }))
      })

      server.middlewares.use('/api/eval-data', (_req, res) => {
        let groups = []
        try {
          const content = fs.readFileSync(evalDataFile, 'utf-8')
          const parsed = YAML.load(content)
          if (Array.isArray(parsed)) {
            groups = parsed.map(g => ({
              conversation_group_id: g.conversation_group_id,
              tag: g.tag || 'eval',
              description: g.description || '',
              turnCount: Array.isArray(g.turns) ? g.turns.length : 0,
              turnIds: Array.isArray(g.turns) ? g.turns.map(t => t.turn_id) : [],
            }))
          }
        } catch { /* ignore */ }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ path: evalDataFile, groups }))
      })

      server.middlewares.use('/api/eval-data-content', (req, res) => {
        if (req.method === 'POST') {
          let body = ''
          req.on('data', c => { body += c })
          req.on('end', () => {
            try {
              const { content } = JSON.parse(body)
              if (!evalDataFile) {
                res.statusCode = 400
                res.setHeader('Content-Type', 'application/json')
                res.end(JSON.stringify({ error: 'LS_EVAL_DATA_PATH not set' }))
                return
              }
              fs.writeFileSync(evalDataFile, content, 'utf-8')
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ ok: true, content }))
            } catch (err) {
              res.statusCode = 500
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ error: err.message }))
            }
          })
          return
        }

        let content = ''
        if (evalDataFile) {
          try {
            content = fs.readFileSync(evalDataFile, 'utf-8')
          } catch { /* ignore */ }
        }
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({ set: !!evalDataFile, path: evalDataFile, content }))
      })

      server.middlewares.use('/api/delete-evaluation', (req, res, next) => {
        if (req.method !== 'POST') return next()
        let body = ''
        req.on('data', c => { body += c })
        req.on('end', () => {
          try {
            const { filename } = JSON.parse(body)
            if (!filename || !/^evaluation_\d{8}_\d{6}_detailed\.csv$/.test(filename)) {
              res.statusCode = 400
              res.setHeader('Content-Type', 'application/json')
              res.end(JSON.stringify({ error: 'Invalid filename' }))
              return
            }
            const tsMatch = filename.match(/(\d{8}_\d{6})/)
            const ts = tsMatch ? tsMatch[1] : null
            const deleted = []

            const csvPath = path.join(evalOutputDir, filename)
            if (fs.existsSync(csvPath)) {
              fs.unlinkSync(csvPath)
              deleted.push(filename)
            }

            if (ts) {
              try {
              for (const f of fs.readdirSync(evalOutputDir)) {
                if (f.includes(ts) && (/\.ya?ml$/.test(f) || /\.json$/.test(f))) {
                  fs.unlinkSync(path.join(evalOutputDir, f))
                  deleted.push(f)
                }
              }
              } catch { /* dir may not exist */ }
              const graphsDir = path.join(evalOutputDir, 'graphs')
              try {
                for (const f of fs.readdirSync(graphsDir)) {
                  if (f.includes(ts)) {
                    fs.unlinkSync(path.join(graphsDir, f))
                    deleted.push(`graphs/${f}`)
                  }
                }
              } catch { /* graphs dir may not exist */ }
            }

            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ ok: true, deleted }))
          } catch (err) {
            res.statusCode = 500
            res.setHeader('Content-Type', 'application/json')
            res.end(JSON.stringify({ error: err.message }))
          }
        })
      })

    },
  }
}

export default defineConfig({
  plugins: [react(), evalDataPlugin()],
  server: {
    fs: { allow: ['.', evalOutputDir] },
  },
})
