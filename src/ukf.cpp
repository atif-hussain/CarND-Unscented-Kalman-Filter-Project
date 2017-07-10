#include "ukf.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 */
UKF::UKF() {
  use_laser_ = true;  /// if this is false, laser measurements will be ignored (except during init)
  use_radar_ = true;  /// if this is false, radar measurements will be ignored (except during init)

  is_initialized_ = false;

  // define state dimensions for ukf model
  n_x_ = 5;		///* State dimension  
  n_aug_ = 7;	///* Augmented state dimension
  n_sig_ = 2 * n_aug_ + 1;  ///* Number of sigma points
  lambda_ = 3.0 - n_aug_; ///* Sigma point spreading parameter
  lambda_n_aug_ = 3.0; ///* Sigma point spreading parameter

  

  // initial state vector	: [pos1 pos2 vel_abs yaw_angle yaw_rate] 
  x_ = VectorXd(5);

  // initial covariance matrix
  P_ = MatrixXd(5, 5);
  P_ << 1, 0, 0, 0, 0,
        0, 1, 0, 0, 0,
        0, 0, 1, 0, 0,
        0, 0, 0, 1, 0,
        0, 0, 0, 0, 1;

  // define Matrices for UKF model
  MatrixXd Xsig_ = MatrixXd(n_x_, n_sig_);  ///* create sigma point matrix  
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_sig_);  ///* create sigma point matrix
  Xsig_pred_ = MatrixXd(n_x_, n_sig_);			  ///* Predicted sigma points as columns

  // Process noise standard deviation longitudinal acceleration in m/s^2
  std_a_ = 0.2;

  // Process noise standard deviation yaw acceleration in rad/s^2
  std_yawdd_ = 0.2;

  // Laser measurement noise standard deviation position1 in m
  std_laspx_ = 0.15;

  // Laser measurement noise standard deviation position2 in m
  std_laspy_ = 0.15;

  // Radar measurement noise standard deviation radius in m
  std_radr_ = 0.3;

  // Radar measurement noise standard deviation angle in rad
  std_radphi_ = 0.03;

  // Radar measurement noise standard deviation radius change in m/s
  std_radrd_ = 0.3;

  // R matrices
  R_radar_ = MatrixXd(3,3);
  R_radar_ << std_radr_*std_radr_, 0, 0,
  0, std_radphi_*std_radphi_, 0,
  0, 0, std_radrd_*std_radrd_;
  
  R_laser_ = MatrixXd(2,2);
  R_laser_ << std_laspx_*std_laspx_, 0,
  0, std_laspy_*std_laspy_;
  
  // Weights of sigma points
  weights_ = VectorXd(n_sig_);
  double weight = 0.5/(lambda_+n_aug_);
  double weight_0 = 2*lambda_*weight;
  weights_(0) = weight_0;
  for (int i=1; i<n_sig_; i++) {
    weights_(i) = weight;
  }

  step_ = 1;
  
  // set small value handling
  p_x_min_ = 0.0001;
  p_y_min_ = 0.0001;

}

UKF::~UKF() {}

/**
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */

static double Mod2PI(double phi) {
  float pi2 = 2*3.14159;
  double r = phi/pi2;
  r -= round(r);
  phi= r*pi2;
  return phi;
}

void UKF::ProcessMeasurement(MeasurementPackage measurement_pack) {
    
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

  if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */
    // first measurement
    cout << "UKF: " << endl;
      
    double px;
    double py;
    double vx;
    double vy;

	/**
	 * Initialize px,py,vx,vy using MeasurementPackage
	*/
    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      /**
      Convert radar from polar to cartesian coordinates and initialize state.
      */
        cout << "init radar" << endl;

        double rho = measurement_pack.raw_measurements_[0];
        double phi = measurement_pack.raw_measurements_[1];
        double rhodot = measurement_pack.raw_measurements_[2];
        px = rho*cos(phi);
        py = rho*sin(phi);
        vx = rhodot*cos(phi);
        vy = rhodot*sin(phi);

    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      /**
      Initialize state.
      */
        cout << "init laser" << endl;

        px = measurement_pack.raw_measurements_[0];
        py = measurement_pack.raw_measurements_[1];
        vx = 0;
        vy = 0;
    }

    // Handle small px, py
    if(fabs(px) < p_x_min_){
        px = p_x_min_;
        cout << "init px too small" << endl;
    }
    if(fabs(py) < p_y_min_){
        py = p_x_min_;
        cout << "init py too small" << endl;
    }


    x_ << px, py, sqrt(pow(vx, 2) + pow(vy, 2)), 0, 0;
    cout << "init x_: " << x_ << endl;

    previous_timestamp_ = measurement_pack.timestamp_;
    // done initializing, no need to predict or update
    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/
    
  cout << "Start predicting" << endl;
  step_ += 1;
  cout << "Step " << step_ << endl;
  double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0; //dt - expressed in seconds
  cout << "dt: " << dt << endl;
  previous_timestamp_ = measurement_pack.timestamp_;

  // Generate Augmented sigma points
  MatrixXd Xsig_aug_ = GenerateAugmentedSigmaPoints();

  // Predict 
  Prediction(Xsig_aug_, dt);
  cout << "End prediction" << endl;

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
    // Radar updates
      cout << "Radar update" << endl;
	  UpdateState(measurement_pack);   
	  cout << "NIS_radar_ = " << NIS_radar_  << endl;
  } else {
    // Laser updates
      cout << "Laser update" << endl;
	  UpdateState(measurement_pack);   
	  cout << "NIS_laser_ = " << NIS_laser_  << endl; 
  }
}

MatrixXd UKF::GenerateAugmentedSigmaPoints() {

  //create augmented mean vector
  VectorXd x_aug = VectorXd(7);

  //create augmented state covariance
  MatrixXd P_aug = MatrixXd(7, 7);
 
  //create augmented mean state
  x_aug.head(5) = x_;
  x_aug(5) = 0;
  x_aug(6) = 0;

  //create augmented covariance matrix
  P_aug.fill(0.0);
  P_aug.topLeftCorner(5,5) = P_;
  P_aug(5,5) = std_a_*std_a_;
  P_aug(6,6) = std_yawdd_*std_yawdd_;

  //create square root matrix
  MatrixXd L = P_aug.llt().matrixL();

  //create augmented sigma points
  MatrixXd Xsig_aug_ = MatrixXd(n_aug_, n_sig_);
  MatrixXd One = MatrixXd(1,n_aug_); One << 1,1,1,1,1,1,1;
  Xsig_aug_ << x_aug, x_aug * One + sqrt(lambda_+n_aug_)*L, x_aug * One - sqrt(lambda_+n_aug_)*L;
  
  //print result
  std::cout << "Xsig_aug_ = " << std::endl << Xsig_aug_ << std::endl;

  return Xsig_aug_;
}

MatrixXd UKF::PredictSigmaPoints(MatrixXd& Xsig_aug_, double delta_t) {

  MatrixXd Xsig_pred = MatrixXd(n_aug_, n_sig_);

  //predict sigma points
  for (int i = 0; i< n_sig_; i++)
  {
    //extract values for better readability
    double p_x = Xsig_aug_(0,i);
    double p_y = Xsig_aug_(1,i);
    double v = Xsig_aug_(2,i);
    double yaw = Xsig_aug_(3,i);
    double yawd = Xsig_aug_(4,i);
    double nu_a = Xsig_aug_(5,i);
    double nu_yawdd = Xsig_aug_(6,i);

    //predicted state values
    double px_p, py_p;

    //avoid division by zero
    if (fabs(yawd) > 0.001) {
        px_p = p_x + v/yawd * ( sin (yaw + yawd*delta_t) - sin(yaw));
        py_p = p_y + v/yawd * ( cos(yaw) - cos(yaw+yawd*delta_t) );
    }
    else {
        px_p = p_x + v*delta_t*cos(yaw);
        py_p = p_y + v*delta_t*sin(yaw);
    }

    double v_p = v;
    double yaw_p = yaw + yawd*delta_t;
    double yawd_p = yawd;

    //add noise
    px_p = px_p + 0.5*nu_a*delta_t*delta_t * cos(yaw);
    py_p = py_p + 0.5*nu_a*delta_t*delta_t * sin(yaw);
    v_p = v_p + nu_a*delta_t;

    yaw_p = yaw_p + 0.5*nu_yawdd*delta_t*delta_t;
    yawd_p = yawd_p + nu_yawdd*delta_t;

    //write predicted sigma point into right column
    Xsig_pred_(0,i) = px_p;
    Xsig_pred_(1,i) = py_p;
    Xsig_pred_(2,i) = v_p;
    Xsig_pred_(3,i) = yaw_p;
    Xsig_pred_(4,i) = yawd_p;
  }

/*******************************************************************************
 * Student part end
 ******************************************************************************/

  //print result
  std::cout << "Xsig_pred_ = " << std::endl << Xsig_pred_ << std::endl;

  return Xsig_pred_;
}

void UKF::PredictMeanAndCovariance(MatrixXd& Xsig_pred_) {

  //create vector for predicted state
  VectorXd x = VectorXd(n_x_);

  //create covariance matrix for prediction
  MatrixXd P = MatrixXd(n_x_, n_x_);

  //predicted state mean
  x.fill(0.0);
  // x = Xsig_pred_ * weights_;
  /* Line above does this:
   */
  for (int i = 0; i < 2 * n_aug_ + 1; i++) {  //iterate over sigma points
    x = x + weights_(i) * Xsig_pred_.col(i);
  }
  // included because array programming is not native to C++
  // but Eigen provides these operations for matrix and vector calcs
  

  //predicted state covariance matrix
  P.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //iterate over sigma points

    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - Xsig_pred_.col(0);
    // VectorXd x_diff = Xsig_pred_.col(i) - x_;

    //angle normalization
    x_diff(3) = Mod2PI(x_diff(3));

    P = P + weights_(i) * x_diff * x_diff.transpose() ;
  }

  x_ = x;
  P_ = P;

  //print result
  std::cout << "Predicted state" << std::endl;
  std::cout << x_ << std::endl;
  std::cout << "Predicted covariance matrix" << std::endl;
  std::cout << P_ << std::endl;

}

/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(MatrixXd& Xsig_aug_, double delta_t) {
  MatrixXd Xsig_pred_ = PredictSigmaPoints(Xsig_aug_, delta_t);
  PredictMeanAndCovariance(Xsig_pred_);
}

void UKF::PredictMeasurement(MeasurementPackage::SensorType sensor) {

  //set measurement dimension, radar can measure r, phi, and r_dot; LIDAR measure px,py
  int n_z = sensor==MeasurementPackage::RADAR? 3: 2;
  
  //matrix for sigma points in measurement space
  Zsig_ = MatrixXd(n_z, n_sig_);
  //transform sigma points into measurement space
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points

    // extract values for better readibility
    double p_x = Xsig_pred_(0,i);
    double p_y = Xsig_pred_(1,i);
    double v  = Xsig_pred_(2,i);
    double yaw = Xsig_pred_(3,i);

    double v1 = cos(yaw)*v;
    double v2 = sin(yaw)*v;

    // RADAR measurement model
    if (sensor==MeasurementPackage::RADAR) {
		if (fabs(p_x) > p_x_min_ || fabs(p_y) > p_y_min_) {
		  Zsig_(0,i) = sqrt(p_x*p_x + p_y*p_y);	 //r
		  Zsig_(1,i) = atan2(p_y,p_x);      	//phi
		  Zsig_(2,i) = (p_x*v1 + p_y*v2 ) / sqrt(p_x*p_x + p_y*p_y);
		} else {
		  // p_x = 0.001;
		  // p_y = 0.001;
		  Zsig_(0,i) = 0.0;   //r
		  Zsig_(1,i) = 0.0;  //phi
		  Zsig_(2,i) = 0.0;
		}
	} else {	// LIDAR measurement model
		if (fabs(p_x) < p_x_min_) p_x = p_x_min_;
		if (fabs(p_y) < p_y_min_) p_y = p_y_min_;
		Zsig_(0,i) = p_x;
		Zsig_(1,i) = p_y;
	}
  }
  std::cout << "Zsig_: " << Zsig_ << std::endl;

  
  //mean predicted measurement
  z_pred_ = VectorXd(n_z);
  z_pred_.fill(0.0);
  for (int i=0; i < n_sig_; i++) {
      z_pred_ = z_pred_ + weights_(i) * Zsig_.col(i);
  }
  std::cout << "z_pred_: " << std::endl << z_pred_ << std::endl;
  
  
  // predicted measurement covariance
  S_ = MatrixXd(n_z, n_z);
  //measurement covariance matrix S
  S_.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;

    //angle normalization
    if (sensor==MeasurementPackage::RADAR)
		z_diff(1) = Mod2PI(z_diff(1));

    S_ = S_ + weights_(i) * z_diff * z_diff.transpose();
  }
  //add measurement noise covariance matrix
  // some people call this Tc
  if (sensor==MeasurementPackage::RADAR)
	S_ = S_ + R_radar_;
  else
	S_ = S_ + R_laser_;
  std::cout << "S_: " << std::endl << S_ << std::endl;
}

/**
 * Updates the state and the state covariance matrix using a laser/radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateState(MeasurementPackage meas_package) {
  /**
  TODO:
  Complete this function! Use lidar/radar data to update the belief about the object's
  position. Modify the state vector, x_, and covariance, P_.
  You'll also need to calculate the lidar/radar NIS.
  */

  // incoming lidar/radar measurement
  int n_z = (meas_package.sensor_type_==MeasurementPackage::RADAR)? 3: 2; // set measurement dimension (radar = 3, lidar = 2)
  z_ = VectorXd(n_z);
  //z_ << meas_package.raw_measurements_[0], meas_package.raw_measurements_[1];
  z_ = meas_package.raw_measurements_;
  PredictMeasurement(meas_package.sensor_type_);

  //create matrix for cross correlation Tc
  MatrixXd Tc = MatrixXd(n_x_, n_z);

  //calculate cross correlation matrix
  Tc.fill(0.0);
  for (int i = 0; i < n_sig_; i++) {  //2n+1 simga points
  
    //residual
    VectorXd z_diff = Zsig_.col(i) - z_pred_;
    // state difference
    VectorXd x_diff = Xsig_pred_.col(i) - x_;
	
	if (meas_package.sensor_type_ == MeasurementPackage::RADAR) {
		//angle normalization for Radar
		x_diff(3) = Mod2PI(x_diff(3));
		z_diff(1) = Mod2PI(z_diff(1));
	}
    Tc = Tc + weights_(i) * x_diff * z_diff.transpose();
  }

  //Kalman gain K;
  MatrixXd K = Tc * S_.inverse();
  //residual
  VectorXd z_diff = z_ - z_pred_;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
	  z_diff(1) = Mod2PI(z_diff(1));

  cout << "S_inv: " << S_.inverse() << endl << "Tc: " << Tc << endl;
  cout << "K: " << K << endl << "z_diff: " << z_diff << endl;

  //update state mean and covariance matrix
  x_ = x_ + K * z_diff;
  P_ = P_ - K*S_*K.transpose();

  // Calculate NIS
  double NIS_ = z_diff.transpose() * S_.inverse() * z_diff;
  if (meas_package.sensor_type_ == MeasurementPackage::RADAR)
		NIS_radar_ = NIS_; 
	else NIS_laser_ = NIS_;

  //print result
  std::cout << "Updated state x: " << std::endl << x_ << std::endl;
  std::cout << "Updated state covariance P: " << std::endl << P_ << std::endl;
}
