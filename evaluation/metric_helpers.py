from six import string_types


def get_class_specific_metrics(class_iterable, metric_iterable):
    class_metric_dict = dict({})
    for class_name, metric_value in zip(class_iterable, metric_iterable):
        class_metric_dict[get_string_type(class_name)] = str(metric_value)
    return class_metric_dict


def get_macro_metrics(metric_value_iterable):
    if len(metric_value_iterable) == 0:
        raise Exception("Length of Metric Iterable is Zero")
    total = sum(metric_value_iterable)
    return total/len(metric_value_iterable)


def get_string_type(class_name):
    if isinstance(class_name, string_types):
        return str(class_name).replace('.',"")
    return str(int(class_name))
