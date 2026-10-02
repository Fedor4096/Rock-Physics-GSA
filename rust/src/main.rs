// Copyright (C) 2023 Fedor Lozovoi
// SPDX-License-Identifier: AGPL-3.0-only
//
// This file is part of Rock-Physics-GSA.
//
// Rock-Physics-GSA is free software: you can redistribute it and/or modify
// it under the terms of the GNU Affero General Public License as published
// by the Free Software Foundation, version 3 of the License.
//
// Rock-Physics-GSA is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU Affero General Public License for more details.
//
// You should have received a copy of the GNU Affero General Public License
// along with Rock-Physics-GSA. If not, see <https://www.gnu.org/licenses/>.

use ndarray::{Array};
use std::{time::Instant, result, io::BufRead};
use nalgebra::{Matrix3, Matrix6};
use rand::Rng;
 
fn main() {
    let start = Instant::now();
    let f: f64 = 0.8;
    let tetha: f64 = 0.0;

    let k_matrix: f64 = 45.0;
    let mu_matrix: f64 = 20.0;
    let v_matrix: f64 =  0.996;
    let a1_matrix: f64 = 1.0;
    let a2_matrix: f64 = 1.0;
    let a3_matrix: f64 = 1.0;
    let range_tetha_matrix: [[f64;3];1] = [[0.0,std::f64::consts::PI,200.0]];
    let range_phi_matrix: [[f64;3];1] = [[0.0,2.0*std::f64::consts::PI,200.0]];

    let k_fluid: f64 = 2.25;
    let mu_fluid: f64 = 0.0;
    let v_fluid: f64 =  1.0 - v_matrix;
    let a1_fluid: f64 = 1000.0;
    let a2_fluid: f64 = 1000.0;
    let a3_fluid: f64 = 1.0;

    let range_tetha_fluid: [[f64;3];3] = [[0.0,1.555,100.0],[1.555,1.59,100.0],[1.59,std::f64::consts::PI,100.0]];
    let range_phi_fluid: [[f64;3];1] = [[0.0,2.0*std::f64::consts::PI,60.0]];

    // Изотропная матрица, анизотропные включения
    let c_matrix: [[[[f64;3];3];3];3] = calculate_c_klmn_from_k_mu(k_matrix, mu_matrix);
    let c_fluid: [[[[f64;3];3];3];3] = calculate_c_klmn_from_k_mu(k_fluid, mu_fluid);

    // print_voigt(convert_full_stiffness_matrix_to_voigt(c_fluid, true));

    // VTI матрица, изотропные включения
    // let c_matrix: [[[[f64;3];3];3];3] = convert_voigt_to_full_stiffness_matrix( [[71.62672, 31.62672,   31.57625,    0.0,        0.0,        0.0],
    //                                                                                      [31.62672,  71.62672,   31.57625,    0.0,        0.0,        0.0],
    //                                                                                      [31.57625,  31.57625,   71.46204,    0.0,        0.0,        0.0],
    //                                                                                      [0.0,       0.0,        0.0,         15.82651,   0.0,        0.0],
    //                                                                                      [0.0,       0.0,        0.0,         0.0,        15.82651,   0.0],
    //                                                                                      [0.0,       0.0,        0.0,         0.0,         0.0,       20.0]], true);
    // let c_fluid: [[[[f64;3];3];3];3] = calculate_c_klmn_from_k_mu(k_fluid, mu_fluid);
    // print_voigt(convert_full_stiffness_matrix_to_voigt(c_fluid, true));
    let c_klmn = tensors_sum(tensor_m_w_num(c_matrix, 1.0-f), tensor_m_w_num(c_fluid, f));
    //println!("{:?}", convert_full_stiffness_matrix_to_voigt(c_klmn, false));

    let (tetha_matrix, phi_matrix, tetha_fluid, phi_fluid) = get_axes(range_tetha_matrix, range_phi_matrix, range_tetha_fluid, range_phi_fluid);
    let (lyambda_inversed_all_matrix, lyambda_inversed_all_fluid) = get_all_inversed_lyambda(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, c_klmn, a1_matrix, a2_matrix, a3_matrix, a1_fluid, a2_fluid, a3_fluid);
    
    let (a_all_integrand_function_matrix, a_all_integrand_function_fluid) = integrand_function_for_all_klmn(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, lyambda_inversed_all_matrix, lyambda_inversed_all_fluid, a1_matrix, a2_matrix, a3_matrix, a1_fluid, a2_fluid, a3_fluid);
    
    let (a_matrix, a_fluid) = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, a_all_integrand_function_matrix, a_all_integrand_function_fluid);
    
    // print_full_tensor(a_fluid);

    let (g_matrix, g_fluid) = tensor_g_calculation_for_all_klmn(a_matrix, a_fluid);

    //print_voigt(convert_full_stiffness_matrix_to_voigt(g_fluid, true));
    // print_full_tensor(a_fluid);

    let c_res = calculat_effective_elastic_properties(g_matrix, g_fluid, c_matrix, c_fluid, c_klmn, v_matrix, v_fluid, tetha);

    //print_voigt(convert_full_stiffness_matrix_to_voigt(c_res, true));

    print_voigt(c_res);

    let elapsed = start.elapsed();
    println!("Programm time: {:?}", elapsed);

    // benchmark();
}

fn print_full_tensor(a: [[[[f64;3];3];3];3]) {
    for k in 0..3 {
        for m in 0..3 {
            for l in 0..3 {
                for n in 0..3 {
                    println!("{} {} {} {} \t {:.8}", k+1,m+1,l+1,n+1, a[k][m][l][n]);
                }
            }   
        }
    }
}

fn print_full_tensor_a_sym(a: [[[[f64;3];3];3];3], a_sym: [[[[f64;3];3];3];3]) {
    println!("{}", a[0][0][1][1]);
    for k in 0..3 {
        for l in 0..3 {
            for n in 0..3 {
                for m in 0..3 {
                    println!("a_{})({},{})({}\t 0/25(a_{}{}{}{} + a_{}{}{}{} + a_{}{}{}{} + a_{}{}{}{})\t= {:.6}",
                    k+1,l+1,n+1,m+1, k+1,l+1,n+1,m+1, m+1,l+1,n+1,k+1, k+1,n+1,l+1,m+1, m+1,n+1,l+1,k+1, a_sym[k][l][n][m]);
                    println!("           \t   {:.6} + {:.6} + {:.6} + {:.6} \n", a[k][l][n][m], a[m][l][n][k], a[k][n][l][m], a[m][n][l][k])
                }
            }   
        }
    }
}

fn print_full_tensor_g(a: [[[[f64;3];3];3];3]) {
    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    println!("g_{}{}{}{} = a_{})({},{})({}\t = {:.6}", i+1,j+1,k+1,l+1, i+1,k+1,l+1,j+1, a[i][k][l][j]);
                }
            }   
        }
    }
}

fn print_voigt(a: [[f64;6];6]) {
    for i in 0..6 {
        for j in 0..6 {
            //println!("{} {} \t {:.6}", i+1,j+1, a[i][j]);
            print!("{:.6}\t", a[i][j])
        }
        print!("\n")
    }
    print!("\n")
}

fn tensors_sum(tensor_1: [[[[f64;3];3];3];3], tensor_2: [[[[f64;3];3];3];3]) -> [[[[f64;3];3];3];3] {
    
    let mut result = [[[[0.0;3];3];3];3];

    for i in 0..3 {
        for j in 0..3 {
            for l in 0..3 {
                for k in 0..3 {
                    result[i][j][l][k] = tensor_1[i][j][l][k] + tensor_2[i][j][l][k];
                }
            }
        }
    }

    return result
}

fn tensors_sub(tensor_1: [[[[f64;3];3];3];3], tensor_2: [[[[f64;3];3];3];3]) -> [[[[f64;3];3];3];3] {
    
    let mut result = [[[[0.0;3];3];3];3];

    for i in 0..3 {
        for j in 0..3 {
            for l in 0..3 {
                for k in 0..3 {
                    result[i][j][l][k] = tensor_1[i][j][l][k] - tensor_2[i][j][l][k];
                }
            }
        }
    }

    return result
}

fn tensor_m_w_num(tensor: [[[[f64;3];3];3];3], number: f64) -> [[[[f64;3];3];3];3] {
    
    let mut result = [[[[0.0;3];3];3];3];
    
    for i in 0..3 {
        for j in 0..3 {
            for l in 0..3 {
                for k in 0..3 {
                    result[i][j][l][k] = tensor[i][j][l][k] * number;
                }
            }
        }
    }
    return result
}

fn m_sum(matrix_1: [[f64;6];6], matrix_2: [[f64;6];6]) -> [[f64;6];6] {
    
    let mut result = [[0.0;6];6];

    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = matrix_1[i][j] + matrix_2[i][j];
        }
    }

    return result
}

fn m_sub(matrix_1: [[f64;6];6], matrix_2: [[f64;6];6]) -> [[f64;6];6] {
    
    let mut result = [[0.0;6];6];

    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = matrix_1[i][j] - matrix_2[i][j];
        }
    }

    return result
}

fn m_m_w_num(matrix: [[f64;6];6], number: f64) -> [[f64;6];6] {
    
    let mut result = [[0.0;6];6];
    
    for i in 0..6 {
        for j in 0..6 {
            result[i][j] = matrix[i][j] * number;
        }
    }
    return result
}

fn m_dot(a: [[f64;6];6], b: [[f64;6];6]) -> [[f64;6];6] {

    let matrix_1 = Matrix6::from(a);
    let matrix_2 = Matrix6::from(b);

    let a_dot_matrix = matrix_1.ad_mul(&matrix_2);

    let mut result = [0.0;36];
    result.copy_from_slice(a_dot_matrix.as_slice());

    let mut split_result = [[0.0;6];6];
    let mut counter: usize = 0;
    for i in 0..6 {
        for j in 0..6 {
            split_result[i][j] = result[counter];
            counter += 1;
        }
    }

    // print!("{:?}", split_result);
    // print_voigt(split_result);

    return split_result;
}

fn convert_voigt_to_full_stiffness_matrix(mut c_voigt: [[f64; 6];6], index: bool) -> [[[[f64;3];3];3];3] {
    
    let mut c_full: [[[[f64;3];3];3];3] = [[[[0.0;3];3];3];3];

    fn full_to_voigt_index(i: usize, j: usize) -> usize {
        if i == j {
            return i
        } else {
            return 6-i-j
        }
    }

    if index == true {
        for i in 0..6 {
            for j in 0..6 {
                if i > 2 {
                    c_voigt[i][j] = c_voigt[i][j]/2.0;
                }
                if j > 2 {
                    c_voigt[i][j] = c_voigt[i][j]/2.0;
                }
            }
        }
    }

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    c_full[i][j][k][l] = c_voigt[full_to_voigt_index(i, j)][full_to_voigt_index(k, l)];
                }
            }
        }
    }

    return c_full
}

fn convert_full_stiffness_matrix_to_voigt(c: [[[[f64;3];3];3];3], index: bool) -> [[f64;6];6] {
    
    let voigt_indices: [[usize; 2];6] = [[0, 0], [1, 1], [2, 2], [1, 2], [2, 0], [0, 1]];
    
    let mut voigt_tensor: [[f64; 6];6] = [[0.0; 6];6];

    let mut k = 0;
    let mut l = 0;
    let mut m = 0;
    let mut n = 0;

    for i in 0..6 {
        for j in 0..6 {
            k = voigt_indices[i][0];
            l = voigt_indices[i][1];

            m = voigt_indices[j][0];
            n = voigt_indices[j][1];
            voigt_tensor[i][j] = c[k][l][m][n];
        }    
    }
        

    if index == true {
        for i in 0..6 {
            for j in 0..6 {
                if i > 2 {
                    voigt_tensor[i][j] = voigt_tensor[i][j] * 2.0;
                }
                if j > 2 {
                    voigt_tensor[i][j] = voigt_tensor[i][j] * 2.0;
                } 
            }
        }
    }

    return voigt_tensor
}

fn calculate_c_klmn_from_k_mu(k: f64, mu:f64) -> [[[[f64;3];3];3];3] {
    
    let mut c_voigt:[[f64; 6];6]  = [[0.0; 6];6];
    
    let lambda: f64 = k - (2.0*mu/3.0);

    c_voigt[0][0] = lambda + (2.0*mu);
    c_voigt[0][1] = lambda;
    c_voigt[0][2] = lambda;
    c_voigt[1][0] = lambda;
    c_voigt[1][1] = lambda + (2.0*mu);
    c_voigt[1][2] = lambda;
    c_voigt[2][0] = lambda;
    c_voigt[2][1] = lambda;
    c_voigt[2][2] = lambda + (2.0*mu);
    c_voigt[3][3] = mu;
    c_voigt[4][4] = mu;
    c_voigt[5][5] = mu;

    let c_full: [[[[f64;3];3];3];3] = convert_voigt_to_full_stiffness_matrix(c_voigt, false);

    return c_full
}

fn get_axes(range_tetha_matrix: [[f64;3];1], range_phi_matrix: [[f64;3];1], range_tetha_fluid: [[f64;3];3], range_phi_fluid: [[f64;3];1]) -> ([f64; 200], [f64; 200], [f64; 300], [f64; 60]) {
    
    let mut tetha_matrix = [0.0; 200];
    let mut phi_matrix = [0.0; 200];

    match Array::linspace(range_tetha_matrix[0][0], range_tetha_matrix[0][1], range_tetha_matrix[0][2] as usize).as_slice() {

        Some(x) => tetha_matrix.copy_from_slice(x),

        None => println!("Cannot use this linspace"),
    }

    match Array::linspace(range_phi_matrix[0][0], range_phi_matrix[0][1], range_phi_matrix[0][2] as usize).as_slice() {

        Some(x) => phi_matrix.copy_from_slice(x),

        None => println!("Cannot use this linspace"),
    }

    ///
    ///
    ///
    
    let mut tetha_fluid = [0.0; 300];
    let mut phi_fluid = [0.0; 60];

    let mut counter: usize = 0;
    for i in 0..3 {
        match Array::linspace(range_tetha_fluid[i][0], range_tetha_fluid[i][1], range_tetha_fluid[i][2] as usize).as_slice() {

            Some(x) => tetha_fluid[counter..counter+range_tetha_fluid[i][2] as usize].copy_from_slice(x),
            None => println!("Cannot use this linspace"),
        }
        counter = counter + range_tetha_fluid[i][2] as usize;
    }
    
    match Array::linspace(range_phi_fluid[0][0], range_phi_fluid[0][1], range_phi_fluid[0][2] as usize).as_slice() {

        Some(x) => phi_fluid.copy_from_slice(x),

        None => println!("Cannot use this linspace"),
    }

    //println!("{:?}", phi_fluid);

    //println!("{:?}", tetha_fluid);

    return (tetha_matrix, phi_matrix, tetha_fluid, phi_fluid)
}

fn lyambda_inversed(c_klmn: [[[[f64;3];3];3];3], tetha: f64, phi: f64, a1: f64, a2: f64, a3: f64) -> [[f64;3];3] {
    
    let mut result = [[0.0;3];3];

    let mut n_all = [0.0;3];
    n_all[0] = tetha.sin()*phi.cos()/a1;
    n_all[1] = tetha.sin()*phi.sin()/a2;
    n_all[2] = tetha.cos()/a3;

    for k in 0..3 {
        for l in 0..3 {
            for m in 0..3 {
                for n in 0..3 {
                    result[k][l] += c_klmn[k][m][l][n] * n_all[m] * n_all[n]
                }
            } 
        } 
    }

    let mut matrix = Matrix3::from(result);

    match matrix.try_inverse() {

        Some(x) => matrix.copy_from(&x),

        None => println!("Cannot inverse matrix"),
    }

    let mut result = [0.0;9];

    result.copy_from_slice(matrix.as_slice());

    let mut split_result: [[f64;3];3] = [[0.0;3];3];

    let mut counter: usize = 0;
    for i in 0..3 {
        for j in 0..3 {
            split_result[i][j] = result[counter];
            counter += 1;
        }
    }

    return split_result;
}

fn get_all_inversed_lyambda(tetha_matrix: [f64;200], phi_matrix: [f64;200],tetha_fluid: [f64;300], phi_fluid: [f64;60], c_klmn: [[[[f64;3];3];3];3], a1_matrix: f64, a2_matrix: f64, a3_matrix: f64, a1_fluid: f64, a2_fluid: f64, a3_fluid: f64) -> (Vec<[[f64; 3]; 3]>, Vec<[[f64; 3]; 3]>) {

    let mut lyambda_inversed_all_matrix = vec![[[0.0; 3];3];200*200];
    let mut lyambda_inversed_all_fluid = vec![[[0.0; 3];3];300*60];

    let mut counter = 0;
    for i in tetha_matrix {
        for j in phi_matrix {
            lyambda_inversed_all_matrix[counter] = lyambda_inversed(c_klmn, i, j, a1_matrix, a2_matrix, a3_matrix);
            counter += 1;
        }
    }

    let mut counter = 0;
    for i in tetha_fluid {
        for j in phi_fluid {
            lyambda_inversed_all_fluid[counter] = lyambda_inversed(c_klmn, i, j, a1_fluid, a2_fluid, a3_fluid);
            counter += 1;
        }
    }

    return (lyambda_inversed_all_matrix, lyambda_inversed_all_fluid);
}

fn calculate_n_mn(tetta: &f64, phi: &f64, a1: f64, a2: f64 ,a3: f64, n: usize, m: usize) -> f64 {
    
    let mut n_all = [0.0;3];
    n_all[0] = tetta.sin() * phi.cos() / a1;
    n_all[1] = tetta.sin() * phi.sin() / a2;
    n_all[2] = tetta.cos() / a3;
    
    return n_all[n] * n_all[m]
}

fn integrand_function_for_single_set_klmn_matrix(k: usize, m: usize, l: usize, n: usize, tetha_matrix: [f64;200], phi_matrix: [f64;200], lyambda_inversed_all_matrix: Vec<[[f64; 3]; 3]>,a1_matrix: f64, a2_matrix: f64, a3_matrix: f64) -> [[f64;200];200] {
    
    let mut result = [[0.0;200];200];

    let mut counter = 0;
    for (c_i, i) in tetha_matrix.iter().enumerate() {
        for (c_j, j) in phi_matrix.iter().enumerate() {
            result[c_i][c_j] = calculate_n_mn(i,j,a1_matrix,a2_matrix,a3_matrix,n,m)*lyambda_inversed_all_matrix[counter][k][l]*i.sin();
            counter += 1;
        }
    }
        
    
    return result
}

fn integrand_function_for_single_set_klmn_fluid(k: usize, m: usize, l: usize, n: usize, tetha_fluid: [f64;300], phi_fluid: [f64;60], lyambda_inversed_all_fluid: Vec<[[f64; 3]; 3]>,a1_fluid: f64, a2_fluid: f64, a3_fluid: f64) -> [[f64;300];60] {
    
    let mut result = [[0.0;300];60];

    let mut counter = 0;
    for (c_i, i) in tetha_fluid.iter().enumerate() {
        for (c_j, j) in phi_fluid.iter().enumerate() {
            result[c_j][c_i] = calculate_n_mn(i,j,a1_fluid,a2_fluid,a3_fluid,n,m)*lyambda_inversed_all_fluid[counter][k][l]*i.sin();
            counter += 1;
        }
    }
        
    return result
}

fn integrand_function_for_all_klmn(tetha_matrix: [f64;200], phi_matrix: [f64;200],tetha_fluid: [f64;300], phi_fluid: [f64;60], lyambda_inversed_all_matrix: Vec<[[f64; 3]; 3]>, lyambda_inversed_all_fluid: Vec<[[f64; 3]; 3]>, a1_matrix: f64, a2_matrix: f64, a3_matrix: f64, a1_fluid: f64, a2_fluid: f64, a3_fluid: f64) -> (Vec<[[f64; 200]; 200]>, Vec<[[f64; 300]; 60]>) {//([[[f64; 60];60];81], [[[f64; 100];60];81]) {

    let mut a_all_integrand_function_matrix = vec![[[0.0; 200];200];81];
    let mut a_all_integrand_function_fluid = vec![[[0.0; 300];60];81];

    let mut counter = 0;
    for k in 0..3 {
        for m in 0..3 {
            for l in 0..3 {
                for n in 0..3 {
                    a_all_integrand_function_matrix[counter] = integrand_function_for_single_set_klmn_matrix(k,m,l,n,tetha_matrix,phi_matrix,lyambda_inversed_all_matrix.clone(),a1_matrix,a2_matrix,a3_matrix);
                    a_all_integrand_function_fluid[counter] = integrand_function_for_single_set_klmn_fluid(k,m,l,n,tetha_fluid,phi_fluid,lyambda_inversed_all_fluid.clone(),a1_fluid,a2_fluid,a3_fluid);
                    counter +=1
                } 
            }
        }
    }

    return (a_all_integrand_function_matrix, a_all_integrand_function_fluid)
}  

fn integral_calculation_by_method_of_medium_rectangles_for_single_klmn_matrix(tetha_matrix: [f64;200],phi_matrix: [f64;200],a_matrix: [[f64; 200]; 200]) -> f64 {
    
    let mut result = 0.0;
    
    let mut mean = 0.0;
    let mut step_x = 0.0;
    let mut step_y = 0.0;
    
    for x in 0..199 {
        for y in 0..199 {
            mean = (a_matrix[x][y] + a_matrix[x+1][y] + a_matrix[x][y+1] + a_matrix[x+1][y+1])/4.0;
            step_x = tetha_matrix[x+1] - tetha_matrix[x];
            step_y = phi_matrix[y+1] - phi_matrix[y];
            result +=  mean*(step_x * step_y)
        } 
    }
        
    result = -1.0/(4.0*std::f64::consts::PI)*result;
    
    return result
}

fn integral_calculation_by_method_of_medium_rectangles_for_single_klmn_fluid(tetha_fluid: [f64;300],phi_fluid: [f64;60],a_fluid: [[f64; 300]; 60]) -> f64 {
    
    let mut result = 0.0;
    
    let mut mean = 0.0;
    let mut step_x = 0.0;
    let mut step_y = 0.0;
    
    for x in 0..299 {
        for y in 0..59 {
            mean = (a_fluid[y][x] + a_fluid[y+1][x] + a_fluid[y][x+1] + a_fluid[y+1][x+1])/4.0;
            step_x = tetha_fluid[x+1] - tetha_fluid[x];
            step_y = phi_fluid[y+1] - phi_fluid[y];
            result +=  mean*(step_x * step_y);
        } 
    }
        
    result = -1.0/(4.0*std::f64::consts::PI)*result;
    
    return result
}

fn integral_calculation_by_method_of_medium_rectangles_for_all(tetha_matrix: [f64;200], phi_matrix: [f64;200],tetha_fluid: [f64;300], phi_fluid: [f64;60], a_matrix: Vec<[[f64; 200]; 200]>, a_fluid: Vec<[[f64; 300]; 60]>) -> ([[[[f64; 3];3];3];3], [[[[f64; 3];3];3];3]) {
    
    let mut a_klmn_all_matrix = [[[[0.0; 3];3];3];3];
    let mut a_klmn_all_fluid = [[[[0.0; 3];3];3];3];

    let mut counter = 0;
    for k in 0..3 {
        for m in 0..3 {
            for l in 0..3 {
                for n in 0..3 {
                    a_klmn_all_matrix[k][m][l][n] = integral_calculation_by_method_of_medium_rectangles_for_single_klmn_matrix(tetha_matrix, phi_matrix, a_matrix[counter]);
                    a_klmn_all_fluid[k][m][l][n] = integral_calculation_by_method_of_medium_rectangles_for_single_klmn_fluid(tetha_fluid, phi_fluid, a_fluid[counter]);
                    counter += 1;
                }
            }   
        }
    }

    return (a_klmn_all_matrix, a_klmn_all_fluid)
}

fn tensor_g_calculation_for_all_klmn(a_matrix: [[[[f64;3];3];3];3], a_fluid: [[[[f64;3];3];3];3]) -> ([[[[f64;3];3];3];3], [[[[f64;3];3];3];3]) {
    
    let mut g_matrix = [[[[0.0;3];3];3];3];
    let mut g_fluid = [[[[0.0;3];3];3];3];

    let mut a_sym_matrix = [[[[0.0;3];3];3];3];
    let mut a_sym_fluid = [[[[0.0;3];3];3];3];


    // *** Без симметризации ***
    // for i in 0..3 {
    //     for j in 0..3 {
    //         for k in 0..3 {
    //             for l in 0..3 {
    //                 g_matrix[i][j][k][l] = 0.25*(a_matrix[i][k][l][j]+a_matrix[j][k][l][i]+a_matrix[i][l][k][j]+a_matrix[j][l][k][i]);
    //                 g_fluid[i][j][k][l] = 0.25*(a_fluid[i][k][l][j]+a_fluid[j][k][l][i]+a_fluid[i][l][k][j]+a_fluid[j][l][k][i]);
    //             }
    //         }
    //     }
    // }
    

    // *** С симметризацией (без шаманства) ***
    for k in 0..3 {
        for l in 0..3 {
            for n in 0..3 {
                for m in 0..3 {
                    a_sym_matrix[k][l][n][m] = 0.25*(a_matrix[k][l][n][m]+a_matrix[m][l][n][k]+a_matrix[k][n][l][m]+a_matrix[m][n][l][k]);
                    a_sym_fluid[k][l][n][m]  = 0.25*(a_fluid[k][l][n][m]+a_fluid[m][l][n][k]+a_fluid[k][n][l][m]+a_fluid[m][n][l][k]);
                }
            }
        }
    }

    // println!("Симметризованный тензор a");
    // print_full_tensor_a_sym(a_matrix, a_sym_matrix);
    // println!("*****************");

    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    g_matrix[i][j][k][l] = a_sym_matrix[i][k][l][j];
                    g_fluid[i][j][k][l] = a_sym_fluid[i][k][l][j];
                }
            }
        }
    }

    // println!("Тензор g после переприсваивания индексов");
    // print_full_tensor_g(a_sym_matrix);
    // println!("*****************\n");

    // *** С симметризацией (с шаманством) ***
    // for k in 0..3 {
    //     for l in 0..3 {
    //         for n in 0..3 {
    //             for m in 0..3 {
    //                 //a_sym_matrix[k][l][n][m] = 0.25*(a_matrix[k][m][l][n]+a_matrix[m][l][n][k]+a_matrix[k][n][l][m]+a_matrix[m][k][n][l]);
    //                 //a_sym_fluid[k][l][n][m] = 0.25*(a_fluid[k][m][l][n]+a_fluid[m][l][n][k]+a_fluid[k][n][l][m]+a_fluid[m][k][n][l]);
    //                 //a_sym_matrix[k][l][n][m] = 0.25*(a_matrix[l][n][k][m]+a_matrix[m][l][n][k]+a_matrix[k][n][l][m]+a_matrix[n][l][m][k]);
    //                 //a_sym_fluid[k][l][n][m] = 0.25*(a_fluid[l][n][k][m]+a_fluid[m][l][n][k]+a_fluid[k][n][l][m]+a_fluid[n][l][m][k]);
    //                 a_sym_matrix[k][l][n][m] = 0.25*(a_matrix[k][m][l][n]+a_matrix[m][l][n][k]+a_matrix[k][n][l][m]+a_matrix[n][l][m][k]);
    //                 a_sym_fluid[k][l][n][m] = 0.25*(a_fluid[k][m][l][n]+a_fluid[m][l][n][k]+a_fluid[k][n][l][m]+a_fluid[n][l][m][k]);
    //             }
    //         }
    //     }
    // }
    // for i in 0..3 {
    //     for k in 0..3 {
    //         for l in 0..3 {
    //             for j in 0..3 {
    //                 g_matrix[i][j][k][l] = a_sym_matrix[i][k][l][j];
    //                 g_fluid[i][j][k][l] = a_sym_fluid[i][k][l][j];
    //             }
    //         }
    //     }
    // }

    // *** С 2-мя компанентами ***
    // for k in 0..3 {
    //     for m in 0..3 {
    //         for l in 0..3 {
    //             for n in 0..3 {
    //                 g_matrix[k][m][l][n] = 0.5*(a_matrix[m][l][n][k]+a_matrix[k][n][l][m]);
    //                 g_fluid[k][m][l][n] = 0.5*(a_fluid[m][l][n][k]+a_fluid[k][n][l][m]);
    //             }
    //         }
    //     }
    // }
    
    return (g_matrix, g_fluid);
} 

fn inv_matrix(a: [[f64;6];6]) -> [[f64;6];6] {
    
    let mut matrix = Matrix6::from(a);

    match matrix.try_inverse() {

        Some(x) => matrix.copy_from(&x),

        None => println!("Cannot inverse matrix"),
    }

    let mut a_voigt_inv = [0.0;36];

    a_voigt_inv.copy_from_slice(matrix.as_slice());

    let mut split_a_voigt_inv = [[0.0;6];6];

    let mut counter: usize = 0;
    for i in 0..6 {
        for j in 0..6 {
            split_a_voigt_inv[i][j] = a_voigt_inv[counter];
            counter += 1;
        }
    }

    return split_a_voigt_inv;

}

fn inverse(a: [[[[f64;3];3];3];3]) -> [[[[f64;3];3];3];3] {
    
    let a_voigt = convert_full_stiffness_matrix_to_voigt(a, true);

    let mut matrix = Matrix6::from(a_voigt);

    match matrix.try_inverse() {

        Some(x) => matrix.copy_from(&x),

        None => println!("Cannot inverse matrix"),
    }

    let mut a_voigt_inv = [0.0;36];

    a_voigt_inv.copy_from_slice(matrix.as_slice());

    let mut split_a_voigt_inv = [[0.0;6];6];

    let mut counter: usize = 0;
    for i in 0..6 {
        for j in 0..6 {
            split_a_voigt_inv[i][j] = a_voigt_inv[counter];
            counter += 1;
        }
    }

    let a_full  = convert_voigt_to_full_stiffness_matrix(split_a_voigt_inv, true);

    return a_full
}

fn multiplication(a: [[[[f64;3];3];3];3], b: [[[[f64;3];3];3];3]) -> [[[[f64;3];3];3];3] {

    let mut result = [[[[0.0;3];3];3];3];


    for i in 0..3 {
        for j in 0..3 {
            for k in 0..3 {
                for l in 0..3 {
                    for m in 0..3 {
                        for n in 0..3 {
                            result[i][j][k][l] += a[i][j][m][n]*b[m][n][k][l]
                        }
                    }
                }
            }
        }
    }
    
    return result;
}

fn cron(i: usize, j: usize) -> f64 {
    if i==j {
        return 1.0;
    } else {
        return 0.0;
    }
}

fn calculat_effective_elastic_properties(g_1: [[[[f64;3];3];3];3], g_2: [[[[f64;3];3];3];3], c_1: [[[[f64;3];3];3];3], c_2: [[[[f64;3];3];3];3], c_klmn: [[[[f64;3];3];3];3], v_1: f64, v_2: f64, tetha: f64) -> [[f64;6];6] {//[[[[f64;3];3];3];3] {
    
    // let mut result = [[[[0.0;3];3];3];3];
    let mut result = [[0.0;6];6];

    // let mut t_1 = [[[[0.0;3];3];3];3];
    // let mut t_2 = [[[[0.0;3];3];3];3];
    // let mut t_3 = [[[[0.0;3];3];3];3];
    // let mut t_4 = [[[[0.0;3];3];3];3];

    let g_1 = convert_full_stiffness_matrix_to_voigt(g_1, true);
    let g_2 = convert_full_stiffness_matrix_to_voigt(g_2, true);
    let c_1 = convert_full_stiffness_matrix_to_voigt(c_1, false);
    let c_2 = convert_full_stiffness_matrix_to_voigt(c_2, false);
    let c_klmn = convert_full_stiffness_matrix_to_voigt(c_klmn, false);

    let t_1_1 = inv_matrix(g_1);
    let t_1_2 = inv_matrix(m_sub(t_1_1, m_sub(c_1, c_klmn)));
    let t_1_3 = m_dot(t_1_2, t_1_1);
    let t_1_4 = m_dot(c_1,t_1_3);
    let t_1 = m_m_w_num(t_1_4, -v_1);

    let t_2_1 = inv_matrix(g_2);
    let t_2_2 = inv_matrix(m_sub(t_2_1, m_sub(c_2, c_klmn)));
    let t_2_3 = m_dot(t_2_2, t_2_1);
    let t_2_4 = m_dot(c_2,t_2_3);
    let t_2 = m_m_w_num(t_2_4, -v_2);

    let t_3_1 = inv_matrix(g_1);
    let t_3_2 = inv_matrix(m_sub(t_3_1, m_sub(c_1, c_klmn)));
    let t_3_3 = m_dot(t_3_2, t_3_1);
    let t_3 = m_m_w_num(t_3_3, -v_1);

    let t_4_1 = inv_matrix(g_2);
    let t_4_2 = inv_matrix(m_sub(t_4_1, m_sub(c_2, c_klmn)));
    let t_4_3 = m_dot(t_4_2, t_4_1);
    let t_4 = m_m_w_num(t_4_3, -v_2);

    // ******************************************
    // t_1 = tensor_m_w_num(multiplication(c_1,multiplication(inverse(tensors_sub(inverse(g_1), tensors_sub(c_1, c_klmn))),inverse(g_1))), -v_1);
    // t_2 = tensor_m_w_num(multiplication(c_2,multiplication(inverse(tensors_sub(inverse(g_2), tensors_sub(c_2, c_klmn))),inverse(g_2))), -v_2);

    // t_3 = tensor_m_w_num(multiplication(inverse(tensors_sub(inverse(g_1), tensors_sub(c_1, c_klmn))),inverse(g_1)), -v_1);
    // t_4 = tensor_m_w_num(multiplication(inverse(tensors_sub(inverse(g_2), tensors_sub(c_2, c_klmn))),inverse(g_2)), -v_2);
    
    result = m_dot(m_sum(t_1, t_2), inv_matrix(m_sum(t_3, t_4)));
    
    return result
}

fn benchmark() {

    let start = Instant::now();

    let mut rng = rand::thread_rng();
    let mut f: f64 = rng.gen_range(0.7..0.9);
    let mut tetha: f64 = 0.0;

    let mut k_matrix: f64 = rng.gen_range(30.0..40.0);
    let mut mu_matrix: f64 = rng.gen_range(15.0..25.0);
    let mut v_matrix: f64 =  rng.gen_range(0.85..0.999);
    let mut a1_matrix: f64 = 1.0;
    let mut a2_matrix: f64 = 1.0;
    let mut a3_matrix: f64 = 1.0;
    let mut range_tetha_matrix: [[f64;3];1] = [[0.0,std::f64::consts::PI,60.0]];
    let mut range_phi_matrix: [[f64;3];1] = [[0.0,2.0*std::f64::consts::PI,60.0]];

    let mut k_fluid: f64 = rng.gen_range(2.0..5.0);
    let mut mu_fluid: f64 = 0.0;
    let mut v_fluid: f64 =  1.0 - v_matrix;
    let mut a1_fluid: f64 = 1000.0;
    let mut a2_fluid: f64 = 1000.0;
    let mut a3_fluid: f64 = 1.0;

    let mut range_tetha_fluid: [[f64;3];3] = [[0.0,1.5,30.0],[1.5,1.64,50.0],[1.64,std::f64::consts::PI,30.0]];
    let mut range_phi_fluid: [[f64;3];1] = [[0.0,2.0*std::f64::consts::PI,60.0]];

    let mut c_matrix: [[[[f64;3];3];3];3] = calculate_c_klmn_from_k_mu(k_matrix, mu_matrix);
    let mut c_fluid: [[[[f64;3];3];3];3] = calculate_c_klmn_from_k_mu(k_fluid, mu_fluid);

    let mut c_klmn = tensors_sum(tensor_m_w_num(c_matrix, 1.0-f), tensor_m_w_num(c_fluid, f));

    let (mut tetha_matrix, mut phi_matrix, mut tetha_fluid, mut phi_fluid) = get_axes(range_tetha_matrix, range_phi_matrix, range_tetha_fluid, range_phi_fluid);
    let (mut lyambda_inversed_all_matrix, mut lyambda_inversed_all_fluid) = get_all_inversed_lyambda(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, c_klmn, a1_matrix, a2_matrix, a3_matrix, a1_fluid, a2_fluid, a3_fluid);
    
    let (mut a_all_integrand_function_matrix, mut a_all_integrand_function_fluid) = integrand_function_for_all_klmn(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, lyambda_inversed_all_matrix, lyambda_inversed_all_fluid, a1_matrix, a2_matrix, a3_matrix, a1_fluid, a2_fluid, a3_fluid);
    
    let (mut a_matrix, mut a_fluid) = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, a_all_integrand_function_matrix, a_all_integrand_function_fluid);
    
    let (mut g_matrix, mut g_fluid) = tensor_g_calculation_for_all_klmn(a_matrix, a_fluid);

    let mut c_res = calculat_effective_elastic_properties(g_matrix, g_fluid, c_matrix, c_fluid, c_klmn, v_matrix, v_fluid, tetha);


    for i in 0..999 {
        f = rng.gen_range(0.7..0.9);
        tetha = 0.0;
    
        k_matrix= rng.gen_range(30.0..40.0);
        mu_matrix= rng.gen_range(15.0..25.0);
        v_matrix =  rng.gen_range(0.85..0.999);
        a1_matrix = 1.0;
        a2_matrix = 1.0;
        a3_matrix = 1.0;
        range_tetha_matrix = [[0.0,std::f64::consts::PI,60.0]];
        range_phi_matrix = [[0.0,2.0*std::f64::consts::PI,60.0]];
    
        k_fluid = rng.gen_range(2.0..5.0);
        mu_fluid = 0.0;
        v_fluid =  1.0 - v_matrix;
        a1_fluid = 1000.0;
        a2_fluid = 1000.0;
        a3_fluid = 1.0;
    
        range_tetha_fluid = [[0.0,1.5,30.0],[1.5,1.64,50.0],[1.64,std::f64::consts::PI,30.0]];
        range_phi_fluid = [[0.0,2.0*std::f64::consts::PI,60.0]];
    
        c_matrix = calculate_c_klmn_from_k_mu(k_matrix, mu_matrix);
        c_fluid = calculate_c_klmn_from_k_mu(k_fluid, mu_fluid);
    
        c_klmn = tensors_sum(tensor_m_w_num(c_matrix, 1.0-f), tensor_m_w_num(c_fluid, f));
    
        (tetha_matrix, phi_matrix, tetha_fluid, phi_fluid) = get_axes(range_tetha_matrix, range_phi_matrix, range_tetha_fluid, range_phi_fluid);
        (lyambda_inversed_all_matrix, lyambda_inversed_all_fluid) = get_all_inversed_lyambda(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, c_klmn, a1_matrix, a2_matrix, a3_matrix, a1_fluid, a2_fluid, a3_fluid);
        
        (a_all_integrand_function_matrix, a_all_integrand_function_fluid) = integrand_function_for_all_klmn(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, lyambda_inversed_all_matrix, lyambda_inversed_all_fluid, a1_matrix, a2_matrix, a3_matrix, a1_fluid, a2_fluid, a3_fluid);
        
        (a_matrix, a_fluid) = integral_calculation_by_method_of_medium_rectangles_for_all(tetha_matrix, phi_matrix, tetha_fluid, phi_fluid, a_all_integrand_function_matrix, a_all_integrand_function_fluid);
        
        (g_matrix, g_fluid) = tensor_g_calculation_for_all_klmn(a_matrix, a_fluid);
    
        c_res = calculat_effective_elastic_properties(g_matrix, g_fluid, c_matrix, c_fluid, c_klmn, v_matrix, v_fluid, tetha);
    }
    let elapsed = start.elapsed();
    println!("Programm time: {:?}", elapsed);
}