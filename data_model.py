import torch
import torch.nn.functional as F
import torch.utils.data as Data
from constant import DIRECTIONS, FILE_SIZES, CUT_OFF, ATOM_NUMBER
from fre_functions import f_c, exponential_map

device = torch.device('cpu')


def generate_radial_samples(distance_a, radial_sample_comb, radial_neighbor_combinations):
    result = []
    dominant_size = radial_sample_comb.size()[0]
    print("generating " + str(dominant_size) + " radial elements")
    for i in range(len(distance_a)):
        neighbor_size = distance_a[i].size()[0]
        neighbor_pairs = radial_neighbor_combinations[neighbor_size]
        rs_list_init = torch.cat(tuple([torch.index_select(distance_a[i], 0, _).unsqueeze(0)
                                        for __, _ in zip(distance_a[i], neighbor_pairs)]))

        rs_list = rs_list_init[:, 1:]
        manipulate_tensor = torch.reshape(torch.cat(
            tuple([rs_list for _ in range(dominant_size)])), (dominant_size, -1))
        # print(manipulate_tensor)
        # print(f_c(manipulate_tensor))
        result.append(exponential_map(
            manipulate_tensor, radial_sample_comb).mul(f_c(manipulate_tensor)).sum(dim=1))

    return result


def generate_angular_samples(distance_a, angular_sample_comb, angular_neighbor_combinations,
                             neighbor_x, neighbor_y, neighbor_z):
    result = []
    dominant_size = angular_sample_comb.size()[0]
    print("generating " + str(dominant_size) + " angular elements")
    for i in range(len(distance_a)):
        neighbor_size = distance_a[i].size()[0]
        neighbor_triples = angular_neighbor_combinations[neighbor_size]
        # print(neighbor_triples)

        # part_1: last_fc
        rs_list_init = torch.cat(tuple([torch.index_select(distance_a[i], 0, _).unsqueeze(0)
                                        for __, _ in zip(neighbor_triples, neighbor_triples)]))
        rs_list = rs_list_init[:, 1:]
        mul_list_1 = f_c(rs_list[:, :1]).mul(f_c(rs_list[:, 1:]))
        manipulate_tensor_1 = torch.reshape(torch.cat(
            tuple([mul_list_1 for _ in range(dominant_size)])), (dominant_size, -1))

        # part_2: exponential
        mul_list_2 = rs_list[:, :1].add(rs_list[:, 1:]).div(2.0)
        manipulate_tensor_2 = torch.reshape(torch.cat(
            tuple([mul_list_2 for _ in range(dominant_size)])), (dominant_size, -1))
        manipulate_tensor_2 = torch.exp(((manipulate_tensor_2 - angular_sample_comb[:, 1:2]) ** 2.0)
                                        .mul(-angular_sample_comb[:, :1])).mul(manipulate_tensor_1)

        # part_3: angular
        x_temp_list = torch.cat(tuple([torch.index_select(neighbor_x[i], 0, _).unsqueeze(0)
                                       for __, _ in zip(neighbor_triples, neighbor_triples)]))
        y_temp_list = torch.cat(tuple([torch.index_select(neighbor_y[i], 0, _).unsqueeze(0)
                                       for __, _ in zip(neighbor_triples, neighbor_triples)]))
        z_temp_list = torch.cat(tuple([torch.index_select(neighbor_z[i], 0, _).unsqueeze(0)
                                       for __, _ in zip(neighbor_triples, neighbor_triples)]))

        x_component_1 = x_temp_list[:, 1:2] - x_temp_list[:, :1]
        y_component_1 = y_temp_list[:, 1:2] - y_temp_list[:, :1]
        z_component_1 = z_temp_list[:, 1:2] - z_temp_list[:, :1]

        x_component_2 = x_temp_list[:, 2:3] - x_temp_list[:, :1]
        y_component_2 = y_temp_list[:, 2:3] - y_temp_list[:, :1]
        z_component_2 = z_temp_list[:, 2:3] - z_temp_list[:, :1]

        inner_product = \
            x_component_1 * x_component_2 + y_component_1 * y_component_2 + z_component_1 * z_component_2

        cosine_triple_angle = inner_product.div(rs_list[:, 0:1]).div(rs_list[:, 1:2])
        sine_triple_angle = torch.sqrt((-cosine_triple_angle ** 2.0).add(1.0))
        manipulate_tensor_3 = torch.reshape(torch.cat(
            tuple([cosine_triple_angle for _ in range(dominant_size)])), (dominant_size, -1))
        manipulate_tensor_4 = torch.reshape(torch.cat(
            tuple([sine_triple_angle for _ in range(dominant_size)])), (dominant_size, -1))
        manipulate_tensor_5 = \
            torch.cos(angular_sample_comb[:, 3:]).mul(
                manipulate_tensor_3).add(torch.sin(angular_sample_comb[:, 3:]).mul(manipulate_tensor_4))
        manipulate_tensor_5 = (manipulate_tensor_5.add(1.0)).pow(
            angular_sample_comb[:, 2:3]).mul(2.0 ** (1 - angular_sample_comb[:, 2:3])).mul(manipulate_tensor_2)

        result.append(manipulate_tensor_5.sum(dim=1))

    return result


def extract_neighbors(x_cat, y_cat, z_cat, neighbor_x, neighbor_y, neighbor_z, distance_a):
    for i in range(ATOM_NUMBER):
        neighbor_x[i].append(torch.reshape(x_cat[0][i], (1,)))
        neighbor_y[i].append(torch.reshape(y_cat[0][i], (1,)))
        neighbor_z[i].append(torch.reshape(z_cat[0][i], (1,)))
        distance_a[i].append(torch.reshape(torch.tensor(0.0, device=device, dtype=torch.float64), (1,)))

    for x_direct, y_direct, z_direct in DIRECTIONS:
        x_cat_temp = x_cat.add(x_direct).clone()
        y_cat_temp = y_cat.add(y_direct).clone()
        z_cat_temp = z_cat.add(z_direct).clone()

        distance = (x_cat.t() - x_cat_temp).pow(2) + (y_cat.t() - y_cat_temp).pow(2) + \
                   (z_cat.t() - z_cat_temp).pow(2)

        position = (torch.le(distance, CUT_OFF ** 2) == 1).nonzero()
        if [x_direct, y_direct, z_direct] == [0, 0, 0]:
            position = position[(position[:, 0] != position[:, 1]).nonzero().squeeze(1)]

        for i in range(ATOM_NUMBER):
            final_position = position[(position[:, 0] == i).nonzero().squeeze(1)][:, 1]
            if final_position.size()[0] == 0:
                continue
            neighbor_x[i].append(torch.index_select(x_cat_temp.t(), 0, final_position)[:, 0])
            neighbor_y[i].append(torch.index_select(y_cat_temp.t(), 0, final_position)[:, 0])
            neighbor_z[i].append(torch.index_select(z_cat_temp.t(), 0, final_position)[:, 0])
            temp_distance = torch.reshape(torch.index_select(distance, 0,
                                                             torch.tensor([i], device=device)), (-1,))
            distance_a[i].append(torch.index_select(temp_distance, 0, final_position).sqrt_())

    for i in range(ATOM_NUMBER):
        neighbor_x[i] = torch.cat(tuple(neighbor_x[i]), dim=0)
        neighbor_y[i] = torch.cat(tuple(neighbor_y[i]), dim=0)
        neighbor_z[i] = torch.cat(tuple(neighbor_z[i]), dim=0)
        distance_a[i] = torch.cat(tuple(distance_a[i]), dim=0)

    return neighbor_x, neighbor_y, neighbor_z, distance_a


def aev_computer(temp_coordinate):
    return


coordinate_temp = torch.load("coordinate03252019.pt")
energy_temp = torch.load("energy03252019.pt")
force_temp = torch.load("force03252019.pt")

torch_data_set = Data.TensorDataset(coordinate_temp, energy_temp, force_temp)
loader = Data.DataLoader(
    dataset=torch_data_set,
    batch_size=64,
    shuffle=True
)

for step, (coordinate, energy, force) in enumerate(loader):
    print(coordinate, energy, force)



