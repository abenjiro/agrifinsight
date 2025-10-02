import { Link } from 'react-router-dom'
import { ArrowRight, Camera, BarChart3, Lightbulb, Shield } from 'lucide-react'

export function HomePage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-primary-50 to-secondary-50">
      {/* Hero Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-20 pb-16">
        <div className="text-center">
          <h1 className="text-4xl md:text-6xl font-bold text-gray-900 mb-6">
            Smart Agriculture with
            <span className="text-primary-600"> AI-Powered</span> Insights
          </h1>
          <p className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto">
            Transform your farming with intelligent disease detection, crop analysis, 
            and personalized recommendations powered by machine learning.
          </p>
          <div className="flex flex-col sm:flex-row gap-4 justify-center">
            <Link
              to="/register"
              className="btn btn-primary px-8 py-3 text-lg rounded-lg inline-flex items-center"
            >
              Get Started
              <ArrowRight className="ml-2 w-5 h-5" />
            </Link>
            <Link
              to="/login"
              className="btn btn-outline px-8 py-3 text-lg rounded-lg"
            >
              Sign In
            </Link>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-16">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Everything you need for modern farming
          </h2>
          <p className="text-lg text-gray-600">
            Our platform combines cutting-edge AI with practical farming knowledge
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
          <div className="card p-6 text-center">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Camera className="w-6 h-6 text-primary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Disease Detection
            </h3>
            <p className="text-gray-600">
              Upload plant images to instantly identify diseases and get treatment recommendations
            </p>
          </div>

          <div className="card p-6 text-center">
            <div className="w-12 h-12 bg-secondary-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <BarChart3 className="w-6 h-6 text-secondary-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Crop Analysis
            </h3>
            <p className="text-gray-600">
              Get detailed insights about your crops' health, growth patterns, and yield predictions
            </p>
          </div>

          <div className="card p-6 text-center">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Lightbulb className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Smart Recommendations
            </h3>
            <p className="text-gray-600">
              Receive personalized advice on planting, fertilizing, and harvesting based on your data
            </p>
          </div>

          <div className="card p-6 text-center">
            <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
              <Shield className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="text-lg font-semibold text-gray-900 mb-2">
              Weather Integration
            </h3>
            <p className="text-gray-600">
              Stay ahead with weather forecasts and climate-based farming recommendations
            </p>
          </div>
        </div>
      </div>

      {/* CTA Section */}
      <div className="bg-primary-600 py-16">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold text-white mb-4">
            Ready to revolutionize your farming?
          </h2>
          <p className="text-xl text-primary-100 mb-8">
            Join thousands of farmers already using AgriFinSight to increase their yields
          </p>
          <Link
            to="/register"
            className="btn bg-white text-primary-600 hover:bg-gray-100 px-8 py-3 text-lg rounded-lg inline-flex items-center"
          >
            Start Your Free Trial
            <ArrowRight className="ml-2 w-5 h-5" />
          </Link>
        </div>
      </div>
    </div>
  )
}

