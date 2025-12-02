import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import {
  BarChart3,
  Download,
  TrendingUp,
  Leaf,
  AlertTriangle,
  Calendar,
  DollarSign,
  Activity,
  FileText,
  Printer,
  Mail,
  ChevronRight,
  Loader,
  RefreshCw,
  Filter,
  Eye
} from 'lucide-react'

interface Farm {
  id: number
  name: string
  size: number
  crops?: any[]
}

interface AnalysisStats {
  total: number
  healthy: number
  diseased: number
  recent: any[]
}

interface CropStats {
  totalCrops: number
  activeCrops: number
  harvestedCrops: number
  byCropType: { [key: string]: number }
}

export function ReportsPage() {
  const navigate = useNavigate()
  const [farms, setFarms] = useState<Farm[]>([])
  const [selectedFarm, setSelectedFarm] = useState<Farm | null>(null)
  const [analysisStats, setAnalysisStats] = useState<AnalysisStats | null>(null)
  const [cropStats, setCropStats] = useState<CropStats | null>(null)
  const [loading, setLoading] = useState(true)
  const [dateRange, setDateRange] = useState('30') // days

  useEffect(() => {
    fetchFarms()
  }, [])

  useEffect(() => {
    if (selectedFarm) {
      fetchReportData()
    }
  }, [selectedFarm, dateRange])

  const fetchFarms = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('auth_token')
      const response = await fetch('http://localhost:8000/api/farms/', {
        headers: {
          'Authorization': `Bearer ${token}`,
          'Content-Type': 'application/json'
        }
      })

      if (!response.ok) throw new Error('Failed to fetch farms')

      const data = await response.json()
      const farmsList = data.data || data
      setFarms(farmsList)

      if (farmsList.length > 0) {
        setSelectedFarm(farmsList[0])
      }
    } catch (err: any) {
      console.error('Error fetching farms:', err)
    } finally {
      setLoading(false)
    }
  }

  const fetchReportData = async () => {
    if (!selectedFarm) return

    try {
      const token = localStorage.getItem('auth_token')

      // Fetch analysis history
      const analysisResponse = await fetch(
        `http://localhost:8000/api/analysis/history?farm_id=${selectedFarm.id}&limit=100`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (analysisResponse.ok) {
        const analysisData = await analysisResponse.json()
        const analyses = analysisData.data || []

        setAnalysisStats({
          total: analyses.length,
          healthy: analyses.filter((a: any) => a.disease_detected?.toLowerCase().includes('healthy')).length,
          diseased: analyses.filter((a: any) => !a.disease_detected?.toLowerCase().includes('healthy')).length,
          recent: analyses.slice(0, 5)
        })
      }

      // Fetch crops for the farm
      const cropsResponse = await fetch(
        `http://localhost:8000/api/farms/${selectedFarm.id}/crops`,
        {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Content-Type': 'application/json'
          }
        }
      )

      if (cropsResponse.ok) {
        const crops = await cropsResponse.json()
        const cropsData = crops.data || crops

        const byCropType: { [key: string]: number } = {}
        cropsData.forEach((crop: any) => {
          byCropType[crop.crop_type] = (byCropType[crop.crop_type] || 0) + 1
        })

        setCropStats({
          totalCrops: cropsData.length,
          activeCrops: cropsData.filter((c: any) => c.is_active).length,
          harvestedCrops: cropsData.filter((c: any) => c.is_harvested).length,
          byCropType
        })
      }
    } catch (err: any) {
      console.error('Error fetching report data:', err)
    }
  }

  const handleExportPDF = () => {
    alert('PDF export functionality coming soon!')
  }

  const handleExportCSV = () => {
    alert('CSV export functionality coming soon!')
  }

  const handleEmailReport = () => {
    alert('Email report functionality coming soon!')
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 w-full">
        <Loader className="h-12 w-12 animate-spin text-green-600" />
        <span className="ml-3 text-gray-600">Loading reports...</span>
      </div>
    )
  }

  if (farms.length === 0) {
    return (
      <div className="w-full max-w-full px-3">
        <div className="bg-white border-0 shadow-soft-xl rounded-2xl p-8 text-center">
          <BarChart3 className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">No Data Available</h3>
          <p className="text-gray-600 mb-6">Create a farm first to generate reports</p>
          <button
            onClick={() => navigate('/farms')}
            className="px-6 py-3 bg-gradient-to-tl from-green-600 to-lime-400 text-white rounded-lg font-semibold hover:scale-102 transition"
          >
            Go to Farms
          </button>
        </div>
      </div>
    )
  }

  return (
    <>
      {/* Header Section */}
      <div className="w-full max-w-full px-3 mb-6">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="p-6">
            <div className="flex flex-wrap items-center justify-between mb-4">
              <div>
                <h6 className="mb-0 font-bold text-gray-900 text-xl">Farm Reports & Analytics</h6>
                <p className="text-sm text-gray-600">Generate comprehensive insights about your farm operations</p>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={fetchReportData}
                  className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 transition text-sm font-medium"
                >
                  <RefreshCw className="w-4 h-4" />
                  Refresh
                </button>
                <button
                  onClick={handleExportPDF}
                  className="flex items-center gap-2 px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition text-sm font-medium"
                >
                  <Printer className="w-4 h-4" />
                  Print
                </button>
                <button
                  onClick={handleExportCSV}
                  className="flex items-center gap-2 px-4 py-2 bg-gradient-to-tl from-green-600 to-lime-400 text-white rounded-lg hover:scale-102 transition text-sm font-medium"
                >
                  <Download className="w-4 h-4" />
                  Export
                </button>
              </div>
            </div>

            {/* Farm Selector and Filters */}
            <div className="flex flex-wrap gap-3 items-center">
              <div className="flex-1 min-w-[200px]">
                <label className="text-xs font-semibold text-gray-600 mb-1 block">Select Farm</label>
                <select
                  value={selectedFarm?.id || ''}
                  onChange={(e) => {
                    const farm = farms.find(f => f.id === Number(e.target.value))
                    setSelectedFarm(farm || null)
                  }}
                  className="w-full px-4 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                >
                  {farms.map((farm) => (
                    <option key={farm.id} value={farm.id}>
                      {farm.name}
                    </option>
                  ))}
                </select>
              </div>

              <div className="w-[180px]">
                <label className="text-xs font-semibold text-gray-600 mb-1 block">Time Period</label>
                <select
                  value={dateRange}
                  onChange={(e) => setDateRange(e.target.value)}
                  className="w-full px-4 py-2 text-sm border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent"
                >
                  <option value="7">Last 7 days</option>
                  <option value="30">Last 30 days</option>
                  <option value="90">Last 3 months</option>
                  <option value="365">Last year</option>
                  <option value="all">All time</option>
                </select>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Quick Stats Row */}
      <div className="w-full max-w-full px-3 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Total Analyses */}
          <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
            <div className="flex-auto p-4">
              <div className="flex items-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-tl from-purple-700 to-pink-500 rounded-lg shadow-soft-md">
                  <Activity className="w-6 h-6 text-white" />
                </div>
                <div className="ml-4">
                  <p className="mb-0 text-sm font-semibold text-gray-600">Total Analyses</p>
                  <h5 className="mb-0 font-bold text-2xl text-gray-900">
                    {analysisStats?.total || 0}
                  </h5>
                </div>
              </div>
            </div>
          </div>

          {/* Healthy Crops */}
          <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
            <div className="flex-auto p-4">
              <div className="flex items-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg shadow-soft-md">
                  <Leaf className="w-6 h-6 text-white" />
                </div>
                <div className="ml-4">
                  <p className="mb-0 text-sm font-semibold text-gray-600">Healthy Detections</p>
                  <h5 className="mb-0 font-bold text-2xl text-gray-900">
                    {analysisStats?.healthy || 0}
                  </h5>
                </div>
              </div>
            </div>
          </div>

          {/* Diseased Detections */}
          <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
            <div className="flex-auto p-4">
              <div className="flex items-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-tl from-red-600 to-rose-400 rounded-lg shadow-soft-md">
                  <AlertTriangle className="w-6 h-6 text-white" />
                </div>
                <div className="ml-4">
                  <p className="mb-0 text-sm font-semibold text-gray-600">Disease Detections</p>
                  <h5 className="mb-0 font-bold text-2xl text-gray-900">
                    {analysisStats?.diseased || 0}
                  </h5>
                </div>
              </div>
            </div>
          </div>

          {/* Active Crops */}
          <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border">
            <div className="flex-auto p-4">
              <div className="flex items-center">
                <div className="flex items-center justify-center w-12 h-12 bg-gradient-to-tl from-blue-600 to-cyan-400 rounded-lg shadow-soft-md">
                  <TrendingUp className="w-6 h-6 text-white" />
                </div>
                <div className="ml-4">
                  <p className="mb-0 text-sm font-semibold text-gray-600">Active Crops</p>
                  <h5 className="mb-0 font-bold text-2xl text-gray-900">
                    {cropStats?.activeCrops || 0}
                  </h5>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Reports Grid */}
      <div className="w-full max-w-full px-3 mb-6 lg:w-8/12 lg:flex-none">
        {/* Crop Health Analysis Report */}
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h6 className="mb-0 font-bold text-gray-900">Crop Health Analysis Report</h6>
                <p className="text-sm text-gray-600">Disease detection summary and trends</p>
              </div>
              <button
                onClick={() => navigate('/analysis')}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-green-700 bg-green-50 hover:bg-green-100 rounded-lg transition"
              >
                View Details
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {analysisStats && analysisStats.total > 0 ? (
              <>
                {/* Health Status Bar */}
                <div className="mb-6">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-gray-600">Health Status Distribution</span>
                    <span className="text-sm font-medium text-gray-900">
                      {analysisStats.total} total scans
                    </span>
                  </div>
                  <div className="w-full h-6 bg-gray-100 rounded-full overflow-hidden flex">
                    <div
                      className="bg-gradient-to-r from-green-500 to-green-400 flex items-center justify-center text-white text-xs font-bold"
                      style={{ width: `${(analysisStats.healthy / analysisStats.total) * 100}%` }}
                    >
                      {analysisStats.healthy > 0 && `${Math.round((analysisStats.healthy / analysisStats.total) * 100)}%`}
                    </div>
                    <div
                      className="bg-gradient-to-r from-red-500 to-red-400 flex items-center justify-center text-white text-xs font-bold"
                      style={{ width: `${(analysisStats.diseased / analysisStats.total) * 100}%` }}
                    >
                      {analysisStats.diseased > 0 && `${Math.round((analysisStats.diseased / analysisStats.total) * 100)}%`}
                    </div>
                  </div>
                  <div className="flex items-center justify-between mt-2">
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                      <span className="text-xs text-gray-600">Healthy: {analysisStats.healthy}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                      <span className="text-xs text-gray-600">Diseased: {analysisStats.diseased}</span>
                    </div>
                  </div>
                </div>

                {/* Recent Analyses */}
                <div>
                  <h6 className="font-semibold text-gray-900 mb-3 text-sm">Recent Analyses</h6>
                  <div className="space-y-2">
                    {analysisStats.recent.slice(0, 5).map((analysis: any, idx: number) => (
                      <div key={idx} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg border border-gray-100">
                        <div className="flex items-center gap-3">
                          <div className={`w-2 h-2 rounded-full ${
                            analysis.disease_detected?.toLowerCase().includes('healthy')
                              ? 'bg-green-500'
                              : 'bg-red-500'
                          }`}></div>
                          <div>
                            <p className="text-sm font-medium text-gray-900">{analysis.disease_detected}</p>
                            <p className="text-xs text-gray-500">
                              {analysis.created_at && new Date(analysis.created_at).toLocaleDateString()}
                            </p>
                          </div>
                        </div>
                        <div className="text-right">
                          <p className="text-sm font-bold text-gray-900">
                            {(analysis.confidence_score * 100).toFixed(0)}%
                          </p>
                          <p className="text-xs text-gray-500">confidence</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              <div className="text-center py-8">
                <Activity className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-600">No analysis data available</p>
              </div>
            )}
          </div>
        </div>

        {/* Crop Production Report */}
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
          <div className="p-6">
            <div className="flex items-center justify-between mb-4">
              <div>
                <h6 className="mb-0 font-bold text-gray-900">Crop Production Report</h6>
                <p className="text-sm text-gray-600">Current crops and harvest status</p>
              </div>
              <button
                onClick={() => navigate('/farms')}
                className="flex items-center gap-2 px-4 py-2 text-sm font-medium text-blue-700 bg-blue-50 hover:bg-blue-100 rounded-lg transition"
              >
                View All Crops
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>

            {cropStats ? (
              <>
                {/* Crop Status Summary */}
                <div className="grid grid-cols-3 gap-4 mb-6">
                  <div className="text-center p-4 bg-blue-50 rounded-lg border border-blue-100">
                    <p className="text-2xl font-bold text-blue-600">{cropStats.totalCrops}</p>
                    <p className="text-xs text-gray-600">Total Crops</p>
                  </div>
                  <div className="text-center p-4 bg-green-50 rounded-lg border border-green-100">
                    <p className="text-2xl font-bold text-green-600">{cropStats.activeCrops}</p>
                    <p className="text-xs text-gray-600">Active</p>
                  </div>
                  <div className="text-center p-4 bg-amber-50 rounded-lg border border-amber-100">
                    <p className="text-2xl font-bold text-amber-600">{cropStats.harvestedCrops}</p>
                    <p className="text-xs text-gray-600">Harvested</p>
                  </div>
                </div>

                {/* Crops by Type */}
                {Object.keys(cropStats.byCropType).length > 0 && (
                  <div>
                    <h6 className="font-semibold text-gray-900 mb-3 text-sm">Crops by Type</h6>
                    <div className="space-y-3">
                      {Object.entries(cropStats.byCropType).map(([cropType, count]) => (
                        <div key={cropType} className="flex items-center justify-between">
                          <div className="flex items-center gap-3 flex-1">
                            <div className="w-8 h-8 bg-gradient-to-tl from-green-600 to-lime-400 rounded-lg flex items-center justify-center">
                              <Leaf className="w-4 h-4 text-white" />
                            </div>
                            <span className="text-sm font-medium text-gray-900">{cropType}</span>
                          </div>
                          <div className="flex items-center gap-4">
                            <div className="w-32 h-2 bg-gray-100 rounded-full overflow-hidden">
                              <div
                                className="h-full bg-gradient-to-r from-green-500 to-lime-400"
                                style={{ width: `${(count / cropStats.totalCrops) * 100}%` }}
                              ></div>
                            </div>
                            <span className="text-sm font-bold text-gray-900 w-8 text-right">{count}</span>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            ) : (
              <div className="text-center py-8">
                <Leaf className="w-12 h-12 text-gray-300 mx-auto mb-3" />
                <p className="text-gray-600">No crop data available</p>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Sidebar - Available Reports */}
      <div className="w-full max-w-full px-3 lg:w-4/12 lg:flex-none">
        <div className="relative flex flex-col min-w-0 break-words bg-white border-0 shadow-soft-xl rounded-2xl bg-clip-border mb-6">
          <div className="p-6">
            <h6 className="mb-4 font-bold text-gray-900">Available Reports</h6>
            <div className="space-y-3">
              {[
                {
                  title: 'Disease Detection Summary',
                  description: 'Comprehensive analysis of detected diseases',
                  icon: AlertTriangle,
                  color: 'from-red-600 to-rose-400',
                  available: true
                },
                {
                  title: 'Crop Yield Forecast',
                  description: 'Predicted yields and harvest timelines',
                  icon: TrendingUp,
                  color: 'from-blue-600 to-cyan-400',
                  available: true
                },
                {
                  title: 'Financial Performance',
                  description: 'Cost analysis and revenue projections',
                  icon: DollarSign,
                  color: 'from-green-600 to-lime-400',
                  available: false
                },
                {
                  title: 'Seasonal Trends',
                  description: 'Historical data and seasonal patterns',
                  icon: Calendar,
                  color: 'from-purple-700 to-pink-500',
                  available: false
                },
                {
                  title: 'Resource Utilization',
                  description: 'Water, fertilizer, and labor usage',
                  icon: Activity,
                  color: 'from-amber-600 to-yellow-400',
                  available: false
                }
              ].map((report, idx) => (
                <div
                  key={idx}
                  className={`p-4 rounded-lg border ${
                    report.available
                      ? 'bg-white border-gray-200 hover:border-green-300 cursor-pointer hover:shadow-md'
                      : 'bg-gray-50 border-gray-100 opacity-60'
                  } transition`}
                >
                  <div className="flex items-start gap-3">
                    <div className={`w-10 h-10 bg-gradient-to-tl ${report.color} rounded-lg flex items-center justify-center flex-shrink-0 shadow-soft-md`}>
                      <report.icon className="w-5 h-5 text-white" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between mb-1">
                        <h6 className="text-sm font-semibold text-gray-900">{report.title}</h6>
                        {report.available && <Eye className="w-4 h-4 text-green-600" />}
                      </div>
                      <p className="text-xs text-gray-600 mb-2">{report.description}</p>
                      {!report.available && (
                        <span className="inline-block px-2 py-1 text-xs font-medium bg-amber-100 text-amber-700 rounded">
                          Coming Soon
                        </span>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Export Options Card */}
        <div className="relative flex flex-col min-w-0 break-words bg-gradient-to-tl from-purple-700 to-pink-500 border-0 shadow-soft-xl rounded-2xl bg-clip-border">
          <div className="flex-auto p-6">
            <div className="text-white">
              <h6 className="mb-2 text-white font-semibold">Export Options</h6>
              <p className="mb-4 text-xs font-normal leading-tight text-white opacity-80">
                Download your reports in various formats
              </p>
              <div className="space-y-2">
                <button
                  onClick={handleExportPDF}
                  className="inline-block w-full px-4 py-2.5 mb-0 font-semibold text-center uppercase align-middle transition-all bg-white rounded-lg cursor-pointer leading-normal text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 text-purple-700"
                >
                  <FileText className="w-3 h-3 inline mr-2" />
                  Export as PDF
                </button>
                <button
                  onClick={handleExportCSV}
                  className="inline-block w-full px-4 py-2.5 mb-0 font-semibold text-center uppercase align-middle transition-all bg-white rounded-lg cursor-pointer leading-normal text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 text-purple-700"
                >
                  <Download className="w-3 h-3 inline mr-2" />
                  Download CSV
                </button>
                <button
                  onClick={handleEmailReport}
                  className="inline-block w-full px-4 py-2.5 mb-0 font-semibold text-center uppercase align-middle transition-all bg-white rounded-lg cursor-pointer leading-normal text-xs ease-soft-in tracking-tight-soft shadow-soft-md hover:scale-102 hover:shadow-soft-xs active:opacity-85 text-purple-700"
                >
                  <Mail className="w-3 h-3 inline mr-2" />
                  Email Report
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  )
}
