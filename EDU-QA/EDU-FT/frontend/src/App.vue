<template>
  <div class="dashboard-container">
    <header class="header">
      <div class="logo">🧬 EduQA 全链路微调与测试平台</div>
      <div class="status-badge" :class="{ 'is-running': isTraining }">
        <span class="dot"></span> {{ isTraining ? '正在训练中...' : '待机状态' }}
      </div>
    </header>

    <div class="main-body">
      <el-tabs v-model="activeTab" class="custom-tabs" @tab-change="handleTabChange">
        
        <el-tab-pane label="⚙️ 训练中枢 (Trainer)" name="train">
          <el-container class="main-layout">
            <el-aside width="450px" class="control-panel">
              <el-card shadow="never" class="config-card">
                <el-form label-position="top" size="small">
                  
                  <el-form-item label="基座模型 (Base Model)">
                    <el-select v-model="form.baseModel" class="full-width">
                      <el-option label="Qwen2.5-7B-Instruct" value="/home/common/hjshen_2025/projects/Qwen/Qwen2.5-7B-Instruct" />
                      <el-option label="Yi-1.5-6B-Chat" value="/home/common/hjshen_2025/.cache/modelscope/hub/models/01ai/Yi-1.5-6B-Chat" />
                    </el-select>
                  </el-form-item>

                  <div class="config-section">
                    <div class="section-title">📚 知识数据源录入 (支持多源合并)</div>
                    
                    <el-form-item label="1. 批量上传 (.jsonl)">
                      <el-upload class="data-upload" drag action="#" :auto-upload="false" :on-change="handleFile" :limit="1">
                        <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                        <div class="el-upload__text">拖拽文件或 <em>点击上传</em></div>
                      </el-upload>
                    </el-form-item>

                    <el-form-item label="2. 动态知识图谱录入 (三元组)">
                      <div class="triple-builder">
                        <el-row :gutter="5" v-for="(t, idx) in manualTriples" :key="idx" style="margin-bottom: 8px;">
                          <el-col :span="7"><el-input v-model="t.sub" placeholder="实体1" /></el-col>
                          <el-col :span="7"><el-input v-model="t.rel" placeholder="关系" /></el-col>
                          <el-col :span="8"><el-input v-model="t.obj" placeholder="实体2" /></el-col>
                          <el-col :span="2"><el-button type="danger" icon="Delete" circle @click="removeTriple(idx)"/></el-col>
                        </el-row>
                        <el-button type="primary" plain size="small" style="width:100%;" @click="addTriple">➕ 添加一条知识记录</el-button>
                      </div>
                    </el-form-item>
                  </div>

                  <div class="config-section">
                    <div class="section-title">⚙️ 优化器与网络架构</div>
                    <el-row :gutter="10">
                      <el-col :span="12"><el-form-item label="批次 (Batch Size)"><el-input-number v-model="form.batchSize" :min="1" class="full-width" /></el-form-item></el-col>
                      <el-col :span="12"><el-form-item label="轮数 (Epochs)"><el-input-number v-model="form.epochs" :min="1" class="full-width"/></el-form-item></el-col>
                    </el-row>
                    <el-row :gutter="10">
                      <el-col :span="12"><el-form-item label="学习率 (LR)"><el-input v-model="form.lr" /></el-form-item></el-col>
                      <el-col :span="12"><el-form-item label="验证集占比(%)"><el-slider v-model="form.valSize" :min="0" :max="20" /></el-form-item></el-col>
                    </el-row>
                  </div>

                  <div style="display: flex; gap: 10px; margin-top: 10px;">
                    <el-button type="primary" class="submit-btn" size="large" @click="startTraining" :loading="isTraining" style="flex: 1;">
                      {{ isTraining ? '微调进行中...' : '开始微调' }}
                    </el-button>
                    <el-button v-if="isTraining" type="danger" plain size="large" @click="cancelTraining">
                      取消
                    </el-button>
                  </div>
                </el-form>
              </el-card>
            </el-aside>

            <el-main class="monitor-panel">
              <el-row :gutter="20" class="metric-row">
                <el-col :span="8">
                  <div class="metric-card">
                    <div class="metric-title">当前进度</div>
                    <el-progress :percentage="progressData.percent" :stroke-width="12" striped striped-flow color="#409eff" />
                  </div>
                </el-col>
                <el-col :span="8">
                  <div class="metric-card">
                    <div class="metric-title">实时 Loss</div>
                    <div class="metric-value loss">{{ progressData.loss || '0.0000' }}</div>
                  </div>
                </el-col>
                <el-col :span="8">
                  <div class="metric-card">
                    <div class="metric-title">训练状态</div>
                    <div class="metric-value" :style="{ color: isTraining ? '#409EFF' : '#67C23A' }">{{ isTraining ? 'Running' : 'Ready' }}</div>
                  </div>
                </el-col>
              </el-row>

              <el-card shadow="never" class="chart-card">
                <template #header><div class="card-title">📉 训练实时Loss曲线 (Loss)</div></template>
                <div ref="lossChartRef" style="height: 220px; width: 100%;"></div>
              </el-card>

              <div class="terminal-container">
                <div class="terminal-header">
                  <span class="mac-btn red"></span><span class="mac-btn yellow"></span><span class="mac-btn green"></span>
                  <span class="terminal-title">bash - LLaMA-Factory GPU Node</span>
                </div>
                <div class="terminal-body" ref="terminalRef">
                  <div v-for="(log, index) in logs" :key="index" class="log-line">
                    <span class="log-time">[{{ log.time }}]</span> <span v-html="log.text"></span>
                  </div>
                  <div v-if="isTraining" class="cursor-blink">_</div>
                </div>
              </div>
            </el-main>
          </el-container>
        </el-tab-pane>

        <el-tab-pane label="💬 对话测试 (Chat)" name="chat">
          <div class="chat-wrapper">
            <div class="chat-header">
              <div style="display: flex; align-items: center; gap: 15px;">
                <span>当前基座：<strong>{{ modelDisplayName }}</strong></span>
                <el-radio-group v-model="chatModelType">
                  <el-radio-button label="base">🧠 原模型</el-radio-button>
                  <el-tooltip :disabled="hasLorAModel" content="暂无微调权重，请先完成训练" placement="top">
                    <span>
                      <el-radio-button label="lora" :disabled="!hasLorAModel">✨ 微调后</el-radio-button>
                    </span>
                  </el-tooltip>
                </el-radio-group>
              </div>
              <el-button type="success" @click="loadModel" :loading="isModelLoading">
                  {{ isModelLoaded ? '重新加载至显存' : '⚡ 确认加载' }}
                </el-button>
                
                <el-button v-if="isModelLoaded" type="warning" plain @click="unloadModel">
                  🧹 释放显存
                </el-button>
            </div>

            <div class="chat-history" ref="chatHistoryRef">
              <div v-for="(msg, idx) in chatMessages" :key="idx" :class="['chat-bubble-wrap', msg.role]">
                <div class="chat-avatar">{{ msg.role === 'user' ? '🧑' : '🤖' }}</div>
                <div class="chat-bubble">{{ msg.content }}</div>
              </div>
            </div>

            <div class="chat-input-area">
              <el-input 
                v-model="chatInput" 
                type="textarea" 
                :rows="3" 
                placeholder="在此输入你要考察的问题... (支持回车发送)" 
                @keyup.enter.exact="sendMessage"
              />
              <el-button type="primary" size="large" @click="sendMessage" :disabled="!isModelLoaded || isGenerating" style="margin-left: 15px;">
                发送 (Send) <el-icon><Position /></el-icon>
              </el-button>
            </div>
          </div>
        </el-tab-pane>

        <el-tab-pane label="📊 定量评测 (Evaluator)" name="eval">
          <el-container class="main-layout">
            <el-aside width="450px" class="control-panel">
              <el-card shadow="never" class="config-card">
                <template #header><div class="card-title"><el-icon><Setting /></el-icon> 性能测试</div></template>
                <el-form label-position="top">
                  <el-form-item label="上传测试集 (.jsonl)">
                    <el-upload class="data-upload" drag action="#" :auto-upload="false" :on-change="handleEvalFile" :limit="1">
                      <el-icon class="el-icon--upload"><UploadFilled /></el-icon>
                      <div class="el-upload__text">拖拽测试集文件至此处</div>
                    </el-upload>
                  </el-form-item>
                  <div class="eval-warning">
                    ⚠️ 提示：测试集较大时，双模型加载与推理将耗费较长时间。
                  </div>
                  <el-button type="primary" size="large" class="submit-btn" @click="startEvaluation" :loading="isEvaluating">
                    {{ isEvaluating ? evalMessage : '🏁 启动微调前后双模型评测' }}
                  </el-button>
                </el-form>
              </el-card>
            </el-aside>
            <el-main class="monitor-panel" style="background: #fff; border-radius: 12px; display: flex; flex-direction: column; align-items: center; justify-content: center;">
              <h3 style="color: #303133; margin-top: 20px;">微调前后模型 性能对比雷达图</h3>
              <div ref="radarChartRef" style="width: 100%; height: 450px;"></div>
              
              <el-table v-if="evalResults" :data="tableData" border style="width: 80%; margin-top: 20px;">
                <el-table-column prop="metric" label="评估指标" width="180" />
                <el-table-column prop="base" label="原模型 (Base)" />
                <el-table-column prop="ft" label="微调后 (Fine-tuned)" />
                <el-table-column prop="diff" label="提升幅度" >
                  <template #default="scope">
                    <span style="color: #67C23A; font-weight: bold;">{{ scope.row.diff }}</span>
                  </template>
                </el-table-column>
              </el-table>
            </el-main>
          </el-container>
        </el-tab-pane>

      </el-tabs>
    </div>
  </div>
</template>

<script setup>
import { ref, reactive, computed, onMounted, nextTick, onUnmounted } from 'vue'
import { Setting, UploadFilled, Delete, Position } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'
import * as echarts from 'echarts'
import axios from 'axios'

const chatModelType = ref('base')
const hasLorAModel = ref(false)
const activeTab = ref('train')

// ========== 训练参数与知识录入 ==========
const form = reactive({ baseModel: '/home/common/hjshen_2025/projects/Qwen/Qwen2.5-7B-Instruct', finetuneType: 'lora', valSize: 5, batchSize: 2, lr: '1e-4', epochs: 5, loraRank: 8, loraAlpha: 16 })
const datasetFile = ref(null)
const manualTriples = ref([{ sub: '', rel: '', obj: '' }])
const modelDisplayName = computed(() => {
  if (!form.baseModel) return ''
  return form.baseModel.split('/').pop()
})
const handleFile = (file) => { datasetFile.value = file.raw }
const addTriple = () => { manualTriples.value.push({ sub: '', rel: '', obj: '' }) }
const removeTriple = (idx) => { manualTriples.value.splice(idx, 1) }

// ========== ECharts 与状态轮询 ==========
const lossChartRef = ref(null)
let lossChartInstance = null
const isTraining = ref(false)
let pollingTimer = null
const terminalRef = ref(null)
let lastLoggedStep = 0; // 🚨 防刷屏记录器

const progressData = reactive({ percent: 0, loss: null, currentEpoch: 0 })
const logs = ref([{ time: new Date().toLocaleTimeString(), text: 'System initialized. Waiting for task configuration...' }])

const pushLog = (text, type = 'info') => {
  let color = { info: '#a6accd', success: '#c3e88d', error: '#f07178', warning: '#ffcb6b' }[type]
  logs.value.push({ time: new Date().toLocaleTimeString(), text: `<span style="color: ${color}">${text}</span>` })
  nextTick(() => { if (terminalRef.value) terminalRef.value.scrollTop = terminalRef.value.scrollHeight })
}

const initChart = () => {
  if (lossChartRef.value && !lossChartInstance) {
    lossChartInstance = echarts.init(lossChartRef.value)
    lossChartInstance.setOption({
      tooltip: { trigger: 'axis' }, grid: { left: '5%', right: '5%', bottom: '15%', top: '10%' },
      xAxis: { type: 'category', data: [], boundaryGap: false }, yAxis: { type: 'value' },
      series: [{ data: [], type: 'line', smooth: true, symbol: 'none', lineStyle: { width: 3, color: '#ff5c5c' },
        areaStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [{ offset: 0, color: 'rgba(255, 92, 92, 0.4)' }, { offset: 1, color: 'rgba(255, 92, 92, 0.0)' }]) }
      }]
    })
  }
}

const pollStatus = async () => {
  if (!isTraining.value) return
  try {
    const res = await axios.get(`http://127.0.0.1:8001/api/training_status?base_model=${form.baseModel}`)
    const data = res.data
    if (data.status === 'training') {
      progressData.percent = data.progress
      progressData.loss = data.current_loss.toFixed(4)
      progressData.currentEpoch = Math.floor((data.progress / 100) * form.epochs) || 1
      
      if (lossChartInstance && data.step.length > 0) {
        lossChartInstance.setOption({ xAxis: { data: data.step }, series: [{ data: data.loss }] })
      }
      
      // 🚨 只有步数增长时才打印日志
      const currentStep = data.step[data.step.length - 1];
      if (currentStep > lastLoggedStep) {
        pushLog(`[GPU Node] Step ${currentStep} | Loss: ${data.current_loss}`, 'info')
        lastLoggedStep = currentStep
      }
      
      if (data.progress >= 100 || data.status === 'completed') {
        finishTraining('success')
      }
    } else if (data.status === 'idle' || data.status === 'completed') {
       finishTraining(data.status === 'completed' ? 'success' : 'cancel')
    }
  } catch (e) { /* ignore network blips */ }
}

const finishTraining = (type) => {
  isTraining.value = false
  clearInterval(pollingTimer)
  if (type === 'success') {
    pushLog('✅ Training completed! Model weights saved successfully.', 'success')
    ElMessage.success('微调完成！快去【对话测试】页面体验吧。')
  } else {
    pushLog('🛑 Training session terminated by user.', 'error')
  }
}

const startTraining = async () => {
  const validTriples = manualTriples.value.filter(t => t.sub && t.rel && t.obj)
  if (!datasetFile.value && validTriples.length === 0) return ElMessage.warning('请上传文件或至少输入一条有效知识！')
  
  isTraining.value = true; progressData.percent = 0; logs.value = []; lastLoggedStep = 0
  pushLog('Packaging hybrid dataset (File + Triples)...', 'warning')
  
  const formData = new FormData()
  if (datasetFile.value) formData.append('file', datasetFile.value)
  formData.append('config', JSON.stringify(form))
  formData.append('manual_triples', JSON.stringify(validTriples))

  try {
    const res = await axios.post('http://127.0.0.1:8001/api/finetune', formData)
    if (res.data.status === 'success') {
      pushLog(res.data.message, 'success')
      pollingTimer = setInterval(pollStatus, 2000)
    }
  } catch (e) {
    isTraining.value = false; pushLog('Failed to connect backend', 'error')
  }
}

// 🚨 改动点：新增取消训练逻辑
const cancelTraining = async () => {
  try {
    const res = await axios.post('http://127.0.0.1:8001/api/cancel_finetune')
    if (res.data.status === 'success') {
      finishTraining('cancel')
      ElMessage.warning('训练已中止')
    }
  } catch (e) {
    ElMessage.error('取消请求失败')
  }
}

// ========== 聊天推理测试 ==========
const isModelLoading = ref(false)
const isModelLoaded = ref(false)
const isGenerating = ref(false)
const chatInput = ref('')
const chatMessages = ref([{ role: 'assistant', content: '你好！我的知识库已更新，请尽情提问我刚才学的知识吧！' }])
const chatHistoryRef = ref(null)

const loadModel = async () => {
  isModelLoading.value = true
  const formData = new FormData()
  formData.append('base_model', form.baseModel)
  formData.append('model_type', chatModelType.value)
  try {
    const res = await axios.post('http://127.0.0.1:8001/api/load_model', formData)
    if (res.data.status === 'success') {
      isModelLoaded.value = true
      ElMessage.success(res.data.message)
    } else {
      ElMessage.error(res.data.message)
    }
  } catch (e) { ElMessage.error('模型加载请求失败') }
  isModelLoading.value = false
}

const unloadModel = async () => {
  try {
    const res = await axios.post('http://127.0.0.1:8001/api/unload_model')
    if (res.data.status === 'success') {
      isModelLoaded.value = false // 状态重置
      // 给对话框发一条系统提示
      chatMessages.value.push({ role: 'assistant', content: '🔌 [系统提示] 模型已从 GPU 卸载，显存已彻底释放。' })
      nextTick(() => { if (chatHistoryRef.value) chatHistoryRef.value.scrollTop = chatHistoryRef.value.scrollHeight })
      
      ElMessage.success(res.data.message)
    } else {
      ElMessage.error(res.data.message)
    }
  } catch (e) { 
    ElMessage.error('释放显存请求失败，请检查网络') 
  }
}

const sendMessage = async () => {
  if (!chatInput.value.trim() || isGenerating.value) return
  chatMessages.value.push({ role: 'user', content: chatInput.value })
  const prompt = chatInput.value
  chatInput.value = ''
  isGenerating.value = true
  
  chatMessages.value.push({ role: 'assistant', content: '思考中...' })
  nextTick(() => { if (chatHistoryRef.value) chatHistoryRef.value.scrollTop = chatHistoryRef.value.scrollHeight })

  try {
    const formData = new FormData(); formData.append('prompt', prompt)
    const res = await axios.post('http://127.0.0.1:8001/api/chat', formData)
    if (res.data.status === 'success') {
      chatMessages.value[chatMessages.value.length - 1].content = res.data.response
    } else {
      chatMessages.value[chatMessages.value.length - 1].content = '出错了：' + res.data.message
    }
  } catch (e) {
    chatMessages.value[chatMessages.value.length - 1].content = '网络请求失败'
  }
  isGenerating.value = false
  nextTick(() => { if (chatHistoryRef.value) chatHistoryRef.value.scrollTop = chatHistoryRef.value.scrollHeight })
}

const handleTabChange = async (name) => {
  if (name === 'train') {
    nextTick(() => { initChart(); if (lossChartInstance) lossChartInstance.resize() })
  }
  if (name === 'chat' || name === 'eval') {
    try {
      const res = await axios.get(`http://127.0.0.1:8001/api/check_model?base_model=${form.baseModel}`)
      hasLorAModel.value = res.data.has_lora
      if (!hasLorAModel.value && chatModelType.value === 'lora') {
        chatModelType.value = 'base'
      }
    } catch (e) { console.error("检查模型状态失败", e) }
  }
}

// ========== 评测与雷达图模块 ==========
const evalDatasetFile = ref(null)
const isEvaluating = ref(false)
const evalMessage = ref('正在准备...')
let evalPollingTimer = null
const radarChartRef = ref(null)
let radarChartInstance = null
const evalResults = ref(null)
const tableData = ref([])

const handleEvalFile = (file) => { evalDatasetFile.value = file.raw }

const initRadarChart = (baseScores, ftScores) => {
  if (radarChartRef.value && !radarChartInstance) {
    radarChartInstance = echarts.init(radarChartRef.value)
  }

  let baseData = [baseScores['BLEU'], baseScores['ROUGE-L'], baseScores['METEOR'] || 0]
  let ftData = [ftScores['BLEU'], ftScores['ROUGE-L'], ftScores['METEOR'] || 0]

  let maxBleu = Math.max(baseData[0], ftData[0]) * 1.1 || 0.01;
  let maxRouge = Math.max(baseData[1], ftData[1]) * 1.1 || 0.01;
  let maxMeteor = Math.max(baseData[2], ftData[2]) * 1.1 || 0.01;

  const option = {
    tooltip: { trigger: 'item' },
    legend: { data: ['Base Model (原模型)', 'Fine-tuned (微调后)'], bottom: 0 },
    radar: {
      indicator: [
        { name: 'BLEU', max: maxBleu },
        { name: 'ROUGE-L', max: maxRouge },
        { name: 'METEOR', max: maxMeteor },
      ],
      shape: 'polygon',
      splitArea: { show: false },
      axisLine: { lineStyle: { color: '#ccc' } },
      splitLine: { lineStyle: { color: '#eee' } },
      axisName: { color: '#fff', backgroundColor: '#666', borderRadius: 3, padding: [3, 5] }
    },
    series: [{
      name: 'Model Comparison',
      type: 'radar',
      data: [
        { value: baseData, name: 'Base Model (原模型)', itemStyle: { color: '#4A90E2' }, areaStyle: { opacity: 0.2, color: '#4A90E2' } },
        { value: ftData, name: 'Fine-tuned (微调后)', itemStyle: { color: '#50E3C2' }, areaStyle: { opacity: 0.4, color: '#50E3C2' } }
      ]
    }]
  }
  radarChartInstance.setOption(option)
}

const pollEvalStatus = async () => {
  try {
    const res = await axios.get('http://127.0.0.1:8001/api/eval_status')
    evalMessage.value = res.data.message
    
    if (res.data.status === 'completed') {
      clearInterval(evalPollingTimer)
      isEvaluating.value = false
      evalResults.value = res.data.results
      
      const calcDiff = (b, f) => {
        if (b === 0 && f > 0) return '从 0 突破'
        return (f - b).toFixed(4)
      }

      tableData.value = [
        { metric: 'BLEU (相似度)', base: res.data.results.base['BLEU'].toFixed(4), ft: res.data.results.ft['BLEU'].toFixed(4), diff: '+' + calcDiff(res.data.results.base['BLEU'], res.data.results.ft['BLEU']) },
        { metric: 'ROUGE-L (召回率)', base: res.data.results.base['ROUGE-L'].toFixed(4), ft: res.data.results.ft['ROUGE-L'].toFixed(4), diff: '+' + calcDiff(res.data.results.base['ROUGE-L'], res.data.results.ft['ROUGE-L']) },
        { metric: 'METEOR (综合匹配)', base: res.data.results.base['METEOR'].toFixed(4), ft: res.data.results.ft['METEOR'].toFixed(4), diff: '+' + calcDiff(res.data.results.base['METEOR'], res.data.results.ft['METEOR']) },
      ]
      
      initRadarChart(res.data.results.base, res.data.results.ft)
      ElMessage.success('评测完成！雷达图已生成。')
    } else if (res.data.status === 'error') {
      clearInterval(evalPollingTimer)
      isEvaluating.value = false
      ElMessage.error(res.data.message)
    }
  } catch (e) { console.error("评测轮询失败", e) }
}

const startEvaluation = async () => {
  if (!evalDatasetFile.value) return ElMessage.warning('请先上传测试集 JSONL 文件！')
  isEvaluating.value = true
  evalResults.value = null
  
  const formData = new FormData()
  formData.append('file', evalDatasetFile.value)
  formData.append('base_model', form.baseModel)

  try {
    const res = await axios.post('http://127.0.0.1:8001/api/start_eval', formData)
    if (res.data.status === 'success') {
      ElMessage.success(res.data.message)
      evalPollingTimer = setInterval(pollEvalStatus, 3000)
    }
  } catch (e) {
    isEvaluating.value = false
    ElMessage.error('无法连接到后端服务器启动评测')
  }
}

onMounted(() => { initChart(); window.addEventListener('resize', () => { if (lossChartInstance) lossChartInstance.resize() }) })
onUnmounted(() => { if (pollingTimer) clearInterval(pollingTimer) })
</script>

<style scoped>
.dashboard-container { min-height: 100vh; background-color: #f0f2f5; }
.header { display: flex; justify-content: space-between; align-items: center; padding: 0 30px; height: 60px; background: #fff; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }
.logo { font-size: 20px; font-weight: 700; background: linear-gradient(90deg, #409eff, #36cfc9); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.main-body { padding: 20px; max-width: 1400px; margin: 0 auto; }
.eval-warning { font-size: 12px; color: #E6A23C; background: #fdf6ec; padding: 10px; border-radius: 6px; margin-bottom: 20px; border: 1px solid #faecd8; }
.custom-tabs :deep(.el-tabs__item) { font-size: 16px; font-weight: bold; }
.main-layout { gap: 20px; height: calc(100vh - 140px); }
.config-section { background: #fafafa; padding: 15px; border-radius: 8px; margin-bottom: 20px; border: 1px solid #ebeef5; }
.section-title { font-size: 13px; font-weight: bold; color: #909399; margin-bottom: 12px; }
.full-width { width: 100%; }
.triple-builder { border: 1px dashed #dcdfe6; padding: 10px; border-radius: 8px; background: #fff; }
.submit-btn { width: 100%; border-radius: 8px; font-size: 16px; background: linear-gradient(90deg, #409eff, #3a8ee6); border: none; }
.monitor-panel { display: flex; flex-direction: column; gap: 15px; padding: 0; overflow: hidden; }
.metric-card { background: #fff; padding: 15px; border-radius: 12px; text-align: center; }
.metric-title { font-size: 14px; color: #909399; margin-bottom: 5px; }
.metric-value { font-size: 24px; font-weight: bold; font-family: monospace; }
.chart-card { border-radius: 12px; margin-bottom: 5px; }
.terminal-container { flex: 1; background: #1e1e1e; border-radius: 12px; display: flex; flex-direction: column; overflow: hidden; border: 1px solid #333; }
.terminal-header { background: #2d2d2d; padding: 10px; display: flex; align-items: center; border-bottom: 1px solid #111; }
.mac-btn { width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
.mac-btn.red { background: #ff5f56; } .mac-btn.yellow { background: #ffbd2e; } .mac-btn.green { background: #27c93f; }
.terminal-title { color: #858585; font-size: 12px; margin: auto; font-family: monospace; }
.terminal-body { padding: 15px; overflow-y: auto; flex: 1; font-family: monospace; font-size: 13px; line-height: 1.5; }
.log-line { margin-bottom: 4px; color: #a6accd; }
.cursor-blink { display: inline-block; width: 6px; height: 12px; background: #fff; animation: blink 1s step-end infinite; }
@keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
.chat-wrapper { background: #fff; border-radius: 12px; height: calc(100vh - 160px); display: flex; flex-direction: column; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
.chat-header { padding: 15px 20px; border-bottom: 1px solid #ebeef5; display: flex; justify-content: space-between; align-items: center; background: #fafafa; border-radius: 12px 12px 0 0; }
.chat-history { flex: 1; padding: 20px; overflow-y: auto; background: #f5f7fa; }
.chat-bubble-wrap { display: flex; margin-bottom: 20px; align-items: flex-start; }
.chat-bubble-wrap.user { flex-direction: row-reverse; }
.chat-avatar { font-size: 24px; margin: 0 15px; }
.chat-bubble { max-width: 60%; padding: 12px 16px; border-radius: 12px; line-height: 1.6; font-size: 15px; word-wrap: break-word; }
.user .chat-bubble { background: #95ec69; color: #333; }
.assistant .chat-bubble { background: #fff; border: 1px solid #ebeef5; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
.chat-input-area { padding: 20px; border-top: 1px solid #ebeef5; display: flex; background: #fff; border-radius: 0 0 12px 12px; }
</style>